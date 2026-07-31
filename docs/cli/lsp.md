# latform-lsp (language server)

`latform-lsp` is a [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
implementation for Bmad lattice files. It brings latform's parsing, formatting,
and linting into your editor as you type.

The server is an optional feature — it depends on
[`pygls`](https://github.com/openlawlibrary/pygls) and is installed with the
`lsp` extra:

```bash
pip install "latform[lsp]"
```

## Features

- **Go to definition / find references** for elements, lines, and constants
- **Hover** with element type and attribute information
- **Completion** of element names, types, and attributes
- **Rename** (with prepare-rename) across the loaded files
- **Formatting** — whole-document and range, using the same formatter as `latform`
- **Document & workspace symbols**
- **Semantic highlighting** (semantic tokens)
- **Code actions**
- **Diagnostics** from the same lint rules as [`latform-lint`](lint.md), including
  the Tao `tao.init` namelist checks (`LF011`–`LF014`)

The server reads your project's `latform.toml` / `pyproject.toml` (see
[Configuration](../configuration.md)) and watches those files, so in-editor
formatting and diagnostics match what the CLI produces. It applies to `*.bmad`,
`*.lat`, and `tao.init` files.

## Running

`latform-lsp` communicates over stdio and is normally launched by an editor, not
by hand:

```
latform-lsp [--log-level {debug,info,warning,error,critical}]
            [--log-file PATH] [--no-client-log] [--stdio]
```

| Option            | Description                                                                             |
| ----------------- | --------------------------------------------------------------------------------------- |
| `--log-level`     | Logging verbosity (default: `warning`, or `$LATFORM_LSP_LOG_LEVEL`)                     |
| `--log-file`      | Write logs to this file instead of stderr (or `$LATFORM_LSP_LOG_FILE`)                  |
| `--no-client-log` | Do not mirror log messages to the client via `window/logMessage`                        |
| `--stdio`         | Communicate over stdio (the default and only transport; accepted if a client passes it) |

Unrecognized flags a client injects (e.g. `--clientProcessId=...`) are ignored.

## Editor plugins

Ready-made clients that discover `latform-lsp` and register the Bmad filetype:

| Editor  | Plugin                                                                      |
| ------- | --------------------------------------------------------------------------- |
| VS Code | [ken-lauer/latform-vscode](https://github.com/ken-lauer/latform-vscode)     |
| Vim     | [ken-lauer/vim-latform](https://github.com/ken-lauer/vim-latform)           |
| Neovim  | [ken-lauer/latform-lsp.nvim](https://github.com/ken-lauer/latform-lsp.nvim) |

Each still needs `latform-lsp` available on `PATH` (`pip install "latform[lsp]"`);
see the plugin's own README for install and configuration details.

## Manual setup

Any LSP client works — point it at the `latform-lsp` command over stdio,
associated with your Bmad files. The examples below are for when you are not
using one of the plugins above.

### Neovim

Using the built-in [LSP client](https://neovim.io/doc/user/lsp.html) (Neovim
0.11+). latform is not in [`nvim-lspconfig`](https://github.com/neovim/nvim-lspconfig)'s
registry, so register it yourself:

```lua
vim.filetype.add({
  extension = { bmad = "bmad", lat = "bmad" },
  filename = { ["tao.init"] = "bmad" },
})

vim.lsp.config("latform", {
  cmd = { "latform-lsp" },
  filetypes = { "bmad" },
  root_markers = { "latform.toml", "pyproject.toml", "tao.init", ".git" },
})
vim.lsp.enable("latform")
```

### Vim

With [`vim-lsp`](https://github.com/prabirshrestha/vim-lsp) (or
[`yegappan/lsp`](https://github.com/yegappan/lsp), or
[`coc.nvim`](https://github.com/neoclide/coc.nvim)):

```vim
augroup latform_ft
  autocmd!
  autocmd BufRead,BufNewFile *.bmad,*.lat,tao.init setfiletype bmad
augroup END

if executable('latform-lsp')
  autocmd User lsp_setup call lsp#register_server({
    \ 'name': 'latform',
    \ 'cmd': {server_info->['latform-lsp']},
    \ 'allowlist': ['bmad'],
    \ })
endif
```

### VS Code

Beyond the [latform-vscode](https://github.com/ken-lauer/latform-vscode)
extension, you can wire the server into any client built on
[`vscode-languageclient`](https://github.com/microsoft/vscode-languageclient),
spawning `latform-lsp` as a stdio server for the `bmad` language.

### Other editors

See the LSP project's
[list of client implementations](https://microsoft.github.io/language-server-protocol/implementors/tools/).
