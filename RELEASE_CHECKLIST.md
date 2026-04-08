# RELEASE CHECKLIST

## Before Publishing
- [x] `public_release_check` contains only allowlisted public files
- [x] `.gitignore` blocks internal data, checkpoints, logs, and local artifacts
- [x] `tools/package_public_release.py` exists for sanitized packaging
- [x] `README.md` explains public status and quick verification commands
- [x] `CHANGELOG.md` summarizes the public baseline

## Public Examples
- [x] `examples/minimal_public_demo.py` exists as a minimal public demonstration

## Verification
- [x] `pytest tests/test_framework_end_to_end.py -q`
- [x] `pytest tests/test_large_evaluator_smoke.py -q`
- [x] `pytest tests/test_framework_generation_reasoning.py -q`
- [x] Public release package verified with no internal files included

## Release Readiness
- [x] Repository published to `https://github.com/rafaelsistems/aksara`
- [ ] Tag a release on GitHub
- [ ] Announce baseline public release
