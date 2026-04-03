# Token-Heavy Commands — Delegate to Codex

These commands can produce large outputs that consume significant Claude context tokens.
When you need to run them, ask Claude to delegate to the Codex subagent instead.

## How to trigger Codex delegation

Tell Claude: **"Use Codex to run `<command>` and summarize the result."**
Or run directly: `! codex "<task description>"`

---

## Command Reference

### Git history / diffs


| Command                       | Why it's heavy                  | Codex task                                                                   |
| ----------------------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| `git diff HEAD~N`             | Full unified diff of many files | "Run `git diff HEAD~5` and summarize files changed + purpose of each change" |
| `git log --stat` or `--patch` | Commit-by-commit file changes   | "Summarize the last 20 commits from `git log --stat`"                        |
| `git log --all --oneline`     | Full branch/tag history         | "List branches and key milestones from `git log --all --oneline`"            |


### Directory / file exploration


| Command          | Why it's heavy              | Codex task                                                                        |
| ---------------- | --------------------------- | --------------------------------------------------------------------------------- |
| `find . -type f` | Entire file tree listing    | "Run `find . -type f -not -path './.git/*'` and group files by directory purpose" |
| `ls -R`          | Recursive directory listing | "List and summarize the directory structure under `data/orion-crc33`"             |
| `tree`           | Same as ls -R               | "Run `tree -L 3` and describe each top-level folder's role"                       |


### Log / output files


| Command                        | Why it's heavy                          | Codex task                                                                   |
| ------------------------------ | --------------------------------------- | ---------------------------------------------------------------------------- |
| `cat <large log>`              | Training logs can be thousands of lines | "Summarize the last 200 lines of `train.log` — loss trend, warnings, errors" |
| `cat <checkpoint dir listing>` | Many checkpoint files                   | "List checkpoints under `outputs/` and identify the latest by step"          |


### Python / config inspection


| Command                                      | Why it's heavy                    | Codex task                                                                            |
| -------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| `python -c "import <module>; print(vars())"` | Module dumps can be huge          | "Summarize the public API of `diffusion/model/nets/multi_group_tme.py`"               |
| Printing full dataset stats                  | HDF5/numpy arrays print verbosely | "Read `metadata/exp_index.hdf5` and summarize tile count, channel names, split sizes" |


### Package / environment


| Command                           | Why it's heavy          | Codex task                                                                  |
| --------------------------------- | ----------------------- | --------------------------------------------------------------------------- |
| `pip list` / `conda list`         | Hundreds of packages    | "Run `pip list` and identify packages relevant to diffusion model training" |
| `pip show <pkg>` across many pkgs | Repeated verbose output | "Check versions of torch, diffusers, controlnet-aux, and transformers"      |


---

## Notes

- Codex runs via OpenAI credits (your Codex subscription), **not Claude tokens**.
- Use Codex for *exploration and summarization*; bring the summary back to Claude for decisions and code changes.
- For quick targeted searches (single file, known class), Claude's built-in Grep/Glob tools are faster and don't need Codex.

