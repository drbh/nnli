## nnli

[![Latest version](https://img.shields.io/crates/v/nnli.svg)](https://crates.io/crates/nnli)

Interactively explore `safetensors` and `onnx` networks in your CLI.

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
nnli print --path <PATH TO MODEL FILE OR MODEL ID>
```

```bash
# if the model is in your HF cache
nnli print --path microsoft/Phi-3-mini-4k-instruct
# when there is more than one revision, specify the revision
nnli print --path microsoft/Phi-3-mini-4k-instruct@d269012bea6fbe38ce7752c8940fea010eea3383
# or the full path
nnli print --path ~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/d269012bea6fbe38ce7752c8940fea010eea3383/
```

This app is a work in progress, and there is a lot of room for improvement on both the code and user experience (UX) fronts.

features

- [x] read onnx models via `candle-onnx`
- [x] display nodes in tui via `ratatui`
- [x] extract and display node details in pane
- [x] improve color schema and ui layout
- [x] improve navigation
- [x] upload to crates.io
- [x] better install instructs
- [ ] build releases
- [ ] improve details output to show all relevant data
- [ ] highligh I/O of node on left
- [ ] add command to show only unique operations
- [ ] add commands to see other useful stats
- [x] support `safetensors` files
- [x] support file or directory input as `--path`
