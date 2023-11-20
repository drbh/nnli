## nnli

[![Latest version](https://img.shields.io/crates/v/nnli.svg)](https://crates.io/crates/nnli)

Interactively explore `onnx` networks in your CLI.

![nnlirestyle](https://github.com/drbh/nnli/assets/9896130/876b476d-349a-450c-afce-52a145e4c04f)

Get `nnli` 🎉

From Cargo

```bash
cargo install nnli
```

From Github 

```bash
git clone https://github.com/drbh/nnli.git
cd nnli
cargo install --path .
```


Check version
```bash
nnli --version
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
- [X] improve color schema and ui layout
- [X] improve navigation
- [X] upload to crates.io
- [X] better install instructs
- [ ] build releases
- [ ] improve details output to show all relevant data
- [ ] highligh I/O of node on left
- [ ] add command to show only unique operations
- [ ] add commands to see other useful stats
