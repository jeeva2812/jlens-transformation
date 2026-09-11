# Attic

Everything here belonged to an arm that is not part of the current write-up.
Nothing in `jlens/`, `tests/`, `verify.py` or `docs/` imports from it. It is kept
because several of these arms have real results recorded in
`docs/MASTER_REPORT_terse.md`, and deleting the code would orphan those numbers.

| path | what |
|---|---|
| `code/` | 196 analysis scripts from the other arms, plus the old `verify_all_arms.py` |
| `docs/` | 30 result notes, briefs and planning docs from those arms |
| `arms/em/` | emergent-misalignment fine-tuning organisms |
| `arms/gauge/` | weight symmetries that leave behaviour exactly unchanged |
| `arms/rh/` | reward hacking and chain-of-thought monitoring |
| `arms/book/` | an earlier long-form write-up generator |
| `arms/upload/` | HuggingFace upload helpers |

To run something from here, restore it first:

```bash
cp _attic/code/<name>.py jlens/
```

Safe to delete in full if you never want these arms back. The results they
produced live in `out/`, not here.
