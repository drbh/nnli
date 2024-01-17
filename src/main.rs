use std::{
    collections::HashMap,
    error::Error,
    io,
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

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Print {
        #[arg(long)]
        path: String,
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

    println!("{buffer:?}");

    // convert into number Python struct.unpack('<Q', buffer)[0]
    let length = u64::from_le_bytes(buffer);

    println!("{length:?}");

    // now read that many bytes of the file
    let mut reader = reader.into_inner();
    let mut buffer = vec![0; length as usize];
    reader.read_exact(&mut buffer)?;

    // the buff is a json string WE NEED TO PERSERVE THE ORDER OF THE LAYERS
    let value: SafetensorFile = serde_json::from_slice(&buffer)?;
    Ok(value)
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

    let loaded = match args.command {
        Command::Print { path: file } => {
            // path is either a file or a directory

            let files = infer_file_types(&file)?;

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
                // load all of the files
                let values = files
                    .iter()
                    .map(|file| read_safetensors_file(file.to_str().unwrap()))
                    .collect::<Result<Vec<_>, _>>()?;

                // merge all of the layers into one
                let mut value = SafetensorFile::default();
                for v in values {
                    value.layers.extend(v.layers);
                }

                // TODO: figure out how to sort the layers to best display the graph
                // value.layers.sort_by(|_key_a, layer_a, _key_b, layer_b| {
                //     layer_a
                //         .data_offsets
                //         .iter()
                //         .sum::<i64>()
                //         .cmp(&layer_b.data_offsets.iter().sum::<i64>())
                // });

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
                println!("File ending not supported");

                // restore terminal
                disable_raw_mode()?;
                execute!(
                    terminal.backend_mut(),
                    LeaveAlternateScreen,
                    DisableMouseCapture
                )?;
                terminal.show_cursor()?;
                return Ok(());
            }
        }
    };

    let res = run_app(&mut terminal, app, tick_rate);

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

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
