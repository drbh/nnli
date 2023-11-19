use std::{
    error::Error,
    io,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*};

use clap::{Parser, Subcommand};

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Print {
        #[arg(long)]
        file: String,
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

use std::collections::HashMap;

// pub fn find_input_nodes(graph: &GraphProto) {

//     // Now, for each node, find its input nodes using the HashMap
//     for node in &graph.node {
//         println!("Node: {}", node.name);
//         for input in &node.input {
//             if let Some(input_nodes) = output_to_input_nodes.get(input) {
//                 println!("Input for {}: {:?}", input, input_nodes);
//             }
//         }
//     }
// }

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
        Command::Print { file } => {
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
                        // input_pointers: input_index_pointers.clone(),
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
        }
    }

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
                format!(" Nodes "),
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
                    .map(|(k, v)| {
                        let line = Line::from(vec![k, v]);
                        line
                    })
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
                    format!(" Details "),
                    Style::default().fg(Color::Black).bg(Color::LightCyan),
                )
                .add_modifier(Modifier::BOLD),
            ),
        )
        .start_corner(Corner::TopLeft);
    f.render_widget(events_list, chunks[1]);
}
