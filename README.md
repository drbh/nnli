## nnli

Interactively explore `onnx` networks in your CLI.

<img width="1140" alt="nnlipoc" src="https://github.com/drbh/nnli/assets/9896130/4bf1d860-bb0b-4622-bb98-d716f11f5d0e">

```
cargo run --release -- print --file <PATH TO ONNX MODEL>
```

This app is a work in progress, and there is a lot of room for improvement on both the code and user experience (UX) fronts. 

features
- [X] read onnx models via `candle-onnx`
- [X] display nodes in tui via `ratatui`
- [X] extract and display node details in pane
- [ ] improve details output to show all relevant data
- [ ] highligh I/O of node on left
- [ ] improve color schema and ui layout
- [ ] improve navigation
- [ ] add command to show only unique operations
- [ ] add commands to see other useful stats
