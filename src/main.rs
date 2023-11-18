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
    events: Vec<(&'a str, String)>,
    graph_data: StatefulList<(String, GraphNode)>,
}

impl<'a> App<'a> {
    fn new() -> App<'a> {
        App {
            events: vec![("Event2", "ERROR".to_string())],
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

            // add graph nodes to app
            let mut graph_data = vec![];
            for node in graph.node {
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

        let info = vec![
            ("Name", gnode.name),
            ("Input", gnode.input.join("\n")),
            ("Output", gnode.output.join("\n")),
            ("OpType", gnode.op_type),
            (
                "Attribute",
                gnode
                    .attribute
                    .iter()
                    .map(|attr| attr.name.clone())
                    .collect::<Vec<String>>()
                    .join("\n"),
            ),
        ];

        let info_str = info
            .iter()
            .map(|(k, v)| format!("{}\n{}", k, v))
            .collect::<Vec<String>>()
            .join("\n");

        app.events.remove(0);
        app.events.insert(0, ("Event1", info_str));
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
        .block(Block::default().borders(Borders::ALL).title("Nodes"))
        .highlight_style(
            Style::default()
                .bg(Color::LightGreen)
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
            // Colorcode the level depending on its type
            let s = match level.as_str() {
                "CRITICAL" => Style::default().fg(Color::Red),
                "ERROR" => Style::default().fg(Color::Magenta),
                "WARNING" => Style::default().fg(Color::Yellow),
                "INFO" => Style::default().fg(Color::Blue),
                _ => Style::default(),
            };
            // Add a example datetime and apply proper spacing between them
            let header = Line::from(vec![
                Span::styled(format!("{level:<9}"), s),
                " ".into(),
                "2020-01-01 10:00:00".italic(),
            ]);
            // The event gets its own line
            // let log = Line::from(vec![event.into()]);

            let log = level
                .split('\n')
                .map(Line::from)
                .collect::<Vec<_>>();

            // Here several things happen:
            // 1. Add a `---` spacing line above the final list entry
            // 2. Add the Level + datetime
            // 3. Add a spacer line
            // 4. Add the actual event

            let mut will_render = vec![
                Line::from("-".repeat(chunks[1].width as usize)),
                header,
                Line::from(""),
            ];

            // join the log lines
            will_render.extend(log);

            ListItem::new(will_render)
        })
        .collect();
    let events_list = List::new(events)
        .block(Block::default().borders(Borders::ALL).title(
            Span::styled(format!("Details"), Style::default().fg(Color::Magenta)),
            // "Details"
        ))
        .start_corner(Corner::BottomLeft);
    f.render_widget(events_list, chunks[1]);
}
