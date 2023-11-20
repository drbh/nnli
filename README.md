## nnli

Interactively explore `onnx` networks in your CLI.

![nnlirestyle](https://github.com/drbh/nnli/assets/9896130/876b476d-349a-450c-afce-52a145e4c04f)

Get `nnli`!

```bash
git clone https://github.com/drbh/nnli.git
cd nnli
cargo install --path .
```

Print a local model

```bash
nnli print --file <PATH TO ONNX MODEL>
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
