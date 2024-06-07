use std::{
    cmp::Ordering,
    collections::HashMap,
    error::Error,
    io,
    str::{self, FromStr},
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use indexmap::IndexMap;
use num_bigint::BigInt;
use ratatui::{prelude::*, widgets::*};
use serde_derive::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetensorFile {
    #[serde(rename = "__metadata__")]
    pub metadata: Metadata,
    #[serde(flatten)]
    pub layers: IndexMap<String, LayerData>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    pub format: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerData {
    pub data_offsets: Vec<i64>,
    pub dtype: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIdDetails {
    org: String,
    model_id: String,
    revision: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]

enum Identifier {
    ModelId(ModelIdDetails),
    Path(String),
}

impl FromStr for Identifier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // This is a bit hacky but makes it easier to work with
        // if contains more than one / then it is a path
        if s.chars().filter(|c| *c == '/').count() > 1 {
            Ok(Identifier::Path(s.to_string()))
        } else {
            let split = s.split('/').collect::<Vec<&str>>();
            let org = split[0].to_string();
            let model_id = split[1].to_string();

            let split = model_id.split('@').collect::<Vec<&str>>();
            let model_id = split[0].to_string();
            let revision = split.get(1).unwrap_or(&"main").to_string();

            Ok(Identifier::ModelId(ModelIdDetails {
                org,
                model_id,
                revision,
            }))
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Print {
        #[arg(long)]
        path: Identifier,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

// Define a simple stateful list widget to store the state of the list
// we will use to display the graph nodes and keep track of the selected node
struct StatefulList<T> {
    state: ListState,
    items: Vec<T>,
}

impl<T> StatefulList<T> {
    fn with_items(items: Vec<T>) -> StatefulList<T> {
        StatefulList {
            state: ListState::default(),
            items,
        }
    }

    fn next(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i >= self.items.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
    }

    fn previous(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i == 0 {
                    self.items.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
    }

    fn unselect(&mut self) {
        self.state.select(None);
    }
}

// A struct to hold the graph node data
#[derive(Clone)]
struct GraphNode {
    name: String,
    input: Vec<String>,
    output: Vec<String>,
    op_type: String,
    attribute: Vec<GraphAttribute>,
    // input_pointers are pointers to the index of the input nodes
    // in the graph_data.items vector
    // input_pointers: Vec<usize>,
}

// A struct to hold the graph node attribute data
#[derive(Clone)]
struct GraphAttribute {
    name: String,
}

/// This struct holds the current state of the app. In particular, it has the `items` field which is
/// a wrapper around `ListState`. Keeping track of the items state let us render the associated
/// widget with its state and have access to features such as natural scrolling.
///
/// Check the event handling at the bottom to see how to change the state on incoming events.
/// Check the drawing logic for items on how to specify the highlighting style for selected items.
struct App<'a> {
    events: Vec<(&'a str, Option<GraphNode>)>,
    graph_data: StatefulList<(String, GraphNode)>,
}

impl<'a> App<'a> {
    fn new() -> App<'a> {
        App {
            events: vec![("Event2", None)],
            graph_data: StatefulList::with_items(vec![]), // Initialize with empty or default data
        }
    }

    /// Rotate through the event list.
    /// This only exists to simulate some kind of "progress"
    fn on_tick(&mut self) {
        // let event = self.events.remove(0);
        // self.events.push(event);
    }

    // Add a method to populate graph data
    fn load_graph_data(&mut self, data: Vec<(String, GraphNode)>) {
        // overwrite the current graph data
        self.graph_data = StatefulList::with_items(data);
    }
}

use std::fs;
use std::path::PathBuf;

fn infer_file_types(file: &str) -> io::Result<Vec<PathBuf>> {
    // Determine if path is a file or a directory
    let metadata = fs::metadata(file)?;
    let files = if metadata.is_dir() {
        // Collect all file paths in the directory
        fs::read_dir(file)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<PathBuf>>()
    } else {
        vec![PathBuf::from(file)]
    };

    // Define allowed extensions
    let allowed_extensions = ["safetensors", "onnx"];

    // Define allowed specific filenames
    let allowed_specific_filenames = ["pytorch_model.bin"];

    // Filter files with allowed extensions or specific filenames
    let allowed_files = files
        .into_iter()
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| allowed_extensions.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
                || allowed_specific_filenames.contains(
                    &path
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default(),
                )
        })
        .collect::<Vec<PathBuf>>();

    Ok(allowed_files)
}

fn read_safetensors_file(file: &str) -> Result<SafetensorFile, Box<dyn Error>> {
    let file = std::fs::File::open(file)?;
    let reader = std::io::BufReader::new(file);

    use std::io::Read;

    // Python f.read(8)
    let mut reader = reader.take(8);
    let mut buffer = [0; 8];
    reader.read_exact(&mut buffer)?;

    // convert into number Python struct.unpack('<Q', buffer)[0]
    let length = u64::from_le_bytes(buffer);

    // now read that many bytes of the file
    let mut reader = reader.into_inner();
    let mut buffer = vec![0; length as usize];
    reader.read_exact(&mut buffer)?;

    // the buff is a json string WE NEED TO PERSERVE THE ORDER OF THE LAYERS
    let value: SafetensorFile = serde_json::from_slice(&buffer)?;
    Ok(value)
}

use std::env;

fn restore_terminal(term: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    // restore terminal
    disable_raw_mode().unwrap();
    execute!(
        term.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .unwrap();
    term.show_cursor().unwrap();
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // create app and run it
    let tick_rate = Duration::from_millis(250);
    let mut app = App::new();

    match args.command {
        Command::Print { path: identifier } => {
            // path is either a file or a directory
            let path = match identifier {
                Identifier::ModelId(model_details) => {
                    let org = model_details.org;
                    let model = model_details.model_id;
                    let mut revision = model_details.revision;

                    let home_path = env::var("HOME").unwrap();
                    let path = PathBuf::from(format!(
                        "{home_path}/.cache/huggingface/hub/models--{org}--{model}/snapshots/"
                    ));

                    // check the one that was last modified
                    let paths = fs::read_dir(path)
                        .map_err(|e| {
                            restore_terminal(&mut terminal);
                            e
                        })?
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .collect::<Vec<PathBuf>>()
                        .clone();

                    if revision == "main" && paths.len() > 1 {
                        restore_terminal(&mut terminal);
                        println!("Select the revision");
                        for path in paths.iter() {
                            let split_on_forward_slash =
                                path.to_str().unwrap().split('/').collect::<Vec<&str>>();
                            let revision = split_on_forward_slash.last().unwrap();
                            println!("add @{} to the model_id", revision);
                        }

                        return Ok(());
                    } else if paths.len() == 1 {
                        // default to the only path
                        revision = paths
                            .first()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .split('/')
                            .last()
                            .unwrap()
                            .to_string();
                    }

                    // filter the paths for the revision
                    let found_path = paths
                        .iter()
                        .find(|path| {
                            path.file_name()
                                .unwrap()
                                .to_str()
                                .unwrap()
                                .contains(&revision)
                        })
                        .unwrap();

                    let path = found_path.to_str().unwrap().to_string();
                    path
                }
                Identifier::Path(path) => path,
            };

            let files = infer_file_types(&path).map_err(|e| {
                restore_terminal(&mut terminal);
                e
            })?;

            let file_ending = files
                .first()
                .unwrap()
                .to_str()
                .unwrap()
                .split('.')
                .last()
                .unwrap();

            if file_ending == "onnx" {
                let file = files.first().unwrap().to_str().unwrap();

                let model = candle_onnx::read_file(file)?;
                // println!("{model:?}");
                let graph = model.graph.unwrap();

                // HashMap to store output-to-input node relationships
                let mut output_to_input_nodes: HashMap<String, usize> = HashMap::new();

                // Iterate through nodes to get the index of the input nodes
                for (i, node) in graph.node.iter().enumerate() {
                    for input in &node.input {
                        output_to_input_nodes.insert(input.clone(), i);
                    }
                }

                // Iterate through nodes to fill the
                // add graph nodes to app
                let mut graph_data = vec![];
                for node in graph.node {
                    // let input_index_pointers = vec![];

                    graph_data.push((
                        node.name.clone(),
                        GraphNode {
                            name: node.name.clone(),
                            input: node.input.clone(),
                            output: node.output.clone(),
                            op_type: node.op_type.clone(),
                            attribute: node
                                .attribute
                                .iter()
                                .map(|attr| GraphAttribute {
                                    name: attr.name.clone(),
                                })
                                .collect(),
                        },
                    ));
                }

                app.load_graph_data(graph_data);
            } else if file_ending == "bin" {
                // read the file
            } else if file_ending == "safetensors" {
                // sort and load all of the files
                let mut sorted_files = files.clone();
                sorted_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

                let values = sorted_files
                    .iter()
                    .map(|file| read_safetensors_file(file.to_str().unwrap()))
                    .collect::<Result<Vec<_>, _>>()?;

                // merge all of the layers into one
                let mut value = SafetensorFile::default();
                for v in values {
                    value.layers.extend(v.layers);
                }

                value
                    .layers
                    .sort_by(|k1, _v1, k2, _v2| cmp_numeric_lexicographic(k1, k2));

                // now populate the graph data
                let mut graph_data = vec![];

                for (name, layer) in value.layers {
                    let est_size = layer.shape.iter().product::<i64>()
                        * match layer.dtype.as_str() {
                            "F8" => 8,
                            "F16" => 2,
                            "F32" => 4,
                            "F64" => 8,
                            "I8" => 1,
                            "I16" => 2,
                            "I32" => 4,
                            "I64" => 8,
                            _ => 0,
                        };

                    let human_readable_size = if est_size <= 1024 {
                        format!("{} B", est_size)
                    } else if est_size <= 1024 * 1024 {
                        format!("{} KB", est_size / 1024)
                    } else if est_size <= 1024 * 1024 * 1024 {
                        format!("{} MB", est_size / 1024 / 1024)
                    } else {
                        format!("{} GB", est_size / 1024 / 1024 / 1024)
                    };

                    graph_data.push((
                        name.clone(),
                        GraphNode {
                            name: name.clone(),
                            input: vec![],
                            output: vec![],
                            op_type: layer.dtype.clone(),
                            attribute: vec![
                                GraphAttribute {
                                    name: format!("shape: {:?}", layer.shape),
                                },
                                GraphAttribute {
                                    name: format!("data_offsets: {:?}", layer.data_offsets),
                                },
                                // estimate the size of the layer by multiplying the shape
                                GraphAttribute {
                                    name: format!("est_weight_size: {}", human_readable_size),
                                },
                            ],
                        },
                    ));
                }

                app.load_graph_data(graph_data);
            } else {
                restore_terminal(&mut terminal);
                println!("File ending not supported");
                return Ok(());
            }
        }
    };

    let res = run_app(&mut terminal, app, tick_rate);
    restore_terminal(&mut terminal);
    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

// This is a helper function to process events and update the events list
fn process_and_update_events(app: &mut App) {
    if let Some(selected_index) = app.graph_data.state.selected() {
        let gnode = app.graph_data.items[selected_index].1.clone();

        // let info = vec![
        //     ("Name", gnode.name),
        //     ("Input", gnode.input.join("\n")),
        //     ("Output", gnode.output.join("\n")),
        //     ("OpType", gnode.op_type),
        //     (
        //         "Attribute",
        //         gnode
        //             .attribute
        //             .iter()
        //             .map(|attr| attr.name.clone())
        //             .collect::<Vec<String>>()
        //             .join("\n"),
        //     ),
        // ];

        // let info_str = info
        //     .iter()
        //     .map(|(k, v)| format!("{}\n{}", k, v))
        //     .collect::<Vec<String>>()
        //     .join("\n");

        app.events.remove(0);
        app.events.insert(0, ("Event1", Some(gnode)));
    }
}

// Run the app and handle events
fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
    tick_rate: Duration,
) -> io::Result<()> {
    let mut last_tick = Instant::now();
    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            return Ok(())
                        }
                        KeyCode::Left => app.graph_data.unselect(),
                        KeyCode::Down => app.graph_data.next(),
                        KeyCode::Up => app.graph_data.previous(),
                        KeyCode::Enter => {} // Additional logic if required
                        _ => {}
                    }

                    process_and_update_events(&mut app); // Call the helper function
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
}

// Draw the app
fn ui(f: &mut Frame, app: &mut App) {
    // Create two chunks with equal horizontal screen space
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(f.size());

    // Iterate through all elements in the `items` app and append some debug text to it.
    let items: Vec<ListItem> = app
        .graph_data
        .items
        .iter()
        .map(|i| {
            let lines = vec![Line::from(i.0.clone())];
            ListItem::new(lines).style(Style::default().fg(Color::White).bg(Color::Black))
        })
        .collect();

    // Create a List from all list items and highlight the currently selected one
    let items = List::new(items)
        .block(
            Block::default().borders(Borders::ALL).title(Span::styled(
                " Nodes ".to_string(),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            )),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    // We can now render the item list
    f.render_stateful_widget(items, chunks[0], &mut app.graph_data.state);

    // Let's do the same for the events.
    // The event list doesn't have any state and only displays the current state of the list.
    let events: Vec<ListItem> = app
        .events
        .iter()
        // .rev()
        .map(|(_event, level)| {
            let line = match level {
                Some(gnode) => {
                    let info = vec![
                        (
                            Span::styled(
                                "Name: ",
                                Style::default()
                                    .fg(Color::Magenta)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(gnode.name.clone(), Style::default().fg(Color::Yellow)),
                        ),
                        (
                            Span::styled(
                                "OpType: ",
                                Style::default()
                                    .fg(Color::Magenta)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(gnode.op_type.clone(), Style::default().fg(Color::Yellow)),
                        ),
                    ];

                    let input_title = vec![(
                        Span::styled("", Style::default().fg(Color::White).bg(Color::Magenta)),
                        Span::styled(
                            format!("Input ({})", gnode.input.len()),
                            Style::default()
                                .fg(Color::Magenta)
                                .add_modifier(Modifier::BOLD),
                        ),
                    )];
                    let inputs = gnode
                        .input
                        .iter()
                        .enumerate()
                        .map(|(index, i)| {
                            if index == gnode.input.len() - 1 {
                                (
                                    Span::styled("  ╰── ", Style::default().fg(Color::White)),
                                    Span::styled(i.clone(), Style::default().fg(Color::Yellow)),
                                )
                            } else {
                                (
                                    Span::styled("  ├── ", Style::default().fg(Color::White)),
                                    Span::styled(i.clone(), Style::default().fg(Color::Yellow)),
                                )
                            }
                        })
                        .collect::<Vec<_>>();

                    let output_title = vec![(
                        Span::styled("", Style::default().fg(Color::White).bg(Color::Magenta)),
                        Span::styled(
                            format!("Output ({})", gnode.output.len()),
                            Style::default()
                                .fg(Color::Magenta)
                                .add_modifier(Modifier::BOLD),
                        ),
                    )];
                    let outputs = gnode
                        .output
                        .iter()
                        .enumerate()
                        .map(|(index, o)| {
                            if index == gnode.output.len() - 1 {
                                (
                                    Span::styled("  ╰── ", Style::default().fg(Color::White)),
                                    Span::styled(o.clone(), Style::default().fg(Color::Yellow)),
                                )
                            } else {
                                (
                                    Span::styled("  ├── ", Style::default().fg(Color::White)),
                                    Span::styled(o.clone(), Style::default().fg(Color::Yellow)),
                                )
                            }
                        })
                        .collect::<Vec<_>>();

                    let attributes_title = vec![(
                        Span::styled("", Style::default().fg(Color::White).bg(Color::Magenta)),
                        Span::styled(
                            format!("Attributes ({})", gnode.output.len()),
                            Style::default()
                                .fg(Color::Magenta)
                                .add_modifier(Modifier::BOLD),
                        ),
                    )];
                    let attributes = gnode
                        .attribute
                        .iter()
                        .enumerate()
                        .map(|(index, a)| {
                            if index == gnode.attribute.len() - 1 {
                                (
                                    Span::styled("  ╰── ", Style::default().fg(Color::White)),
                                    Span::styled(
                                        a.name.clone(),
                                        Style::default().fg(Color::Yellow),
                                    ),
                                )
                            } else {
                                (
                                    Span::styled("  ├── ", Style::default().fg(Color::White)),
                                    Span::styled(
                                        a.name.clone(),
                                        Style::default().fg(Color::Yellow),
                                    ),
                                )
                            }
                        })
                        .collect::<Vec<_>>();

                    let info_lines = vec![(
                        Span::styled("", Style::default().fg(Color::White).bg(Color::Magenta)),
                        Span::styled("".to_string(), Style::default().fg(Color::Yellow)),
                    )]
                    .into_iter()
                    .chain(info)
                    .chain(input_title)
                    .chain(inputs)
                    .chain(output_title)
                    .chain(outputs)
                    .chain(attributes_title)
                    .chain(attributes)
                    .map(|(k, v)| Line::from(vec![k, v]))
                    .collect::<Vec<_>>();

                    ListItem::new(info_lines)
                }
                None => ListItem::new(vec![Line::from("None")]),
            };
            line
        })
        .collect();
    let events_list = List::new(events)
        .block(
            Block::default().borders(Borders::ALL).title(
                Span::styled(
                    " Details ".to_string(),
                    Style::default().fg(Color::Black).bg(Color::LightCyan),
                )
                .add_modifier(Modifier::BOLD),
            ),
        )
        .start_corner(Corner::TopLeft);
    f.render_widget(events_list, chunks[1]);
}

fn cmp_numeric_lexicographic(s1: &str, s2: &str) -> Ordering {
    let mut b1 = s1.as_bytes();
    let mut b2 = s2.as_bytes();

    while !b1.is_empty() && !b2.is_empty() {
        if b1[0].is_ascii_digit() && b2[0].is_ascii_digit() {
            // Do a numerical compare if we encounter some digits.
            let b1_digits = count_digit_bytes(b1);
            let b2_digits = count_digit_bytes(b2);

            // Unwraps are safe. A run of ASCII digits is always valid
            // UTF-8 and always a valid number.
            let num1 = BigInt::from_str(str::from_utf8(&b1[..b1_digits]).unwrap()).unwrap();
            let num2 = BigInt::from_str(str::from_utf8(&b2[..b2_digits]).unwrap()).unwrap();

            match num1.cmp(&num2) {
                Ordering::Equal => {
                    b1 = &b1[b1_digits..];
                    b2 = &b2[b2_digits..];
                }
                ord => return ord,
            }
        } else {
            // If the byte is not a digit, do a lexicographical compare.
            match b1[0].cmp(&b2[0]) {
                Ordering::Equal => {
                    b1 = &b1[1..];
                    b2 = &b2[1..];
                }
                ord => return ord,
            }
        }
    }

    b1.cmp(b2)
}

fn count_digit_bytes(b: &[u8]) -> usize {
    b.iter().take_while(|b| b.is_ascii_digit()).count()
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use crate::cmp_numeric_lexicographic;

    #[test]
    fn test_cmp_lexicographic_numeric() {
        assert_eq!(cmp_numeric_lexicographic("aaa", "aaa"), Ordering::Equal);
        assert_eq!(cmp_numeric_lexicographic("aaa", "aa"), Ordering::Greater);
        assert_eq!(cmp_numeric_lexicographic("aa", "aaa"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("aaa", "aab"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("baa", "aaa"), Ordering::Greater);
        assert_eq!(cmp_numeric_lexicographic("aaa1", "aaa2"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("1aaa", "2aaa"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("aaa1a", "aaa2a"), Ordering::Less);
        assert_eq!(cmp_numeric_lexicographic("aaa1a", "aaa11a"), Ordering::Less);
        assert_eq!(
            cmp_numeric_lexicographic("aaa2a", "aaa1a"),
            Ordering::Greater
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa11a", "aaa1a"),
            Ordering::Greater
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa11a", "aaa011a"),
            Ordering::Equal
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa1abb1b", "aaa1abb1b"),
            Ordering::Equal
        );
        assert_eq!(
            cmp_numeric_lexicographic("aaa1abb1b", "aaa1abb12b"),
            Ordering::Less
        );
    }
}
