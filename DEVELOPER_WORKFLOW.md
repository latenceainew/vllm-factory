# Developer Workflow

Solo-engineer workflow for developing, releasing, and maintaining vllm-factory.

## Day-to-day development

```
main branch = always shippable
```

1. Create a feature branch: `git checkout -b feat/my-change`
2. Make changes, commit often
3. Push and open a PR: `git push -u origin HEAD && gh pr create`
4. CI runs lint + smoke tests automatically
5. Squash-merge to main via GitHub UI (or `gh pr merge --squash`)

## Release checklist

Copy-paste this when you're ready to ship a new version.

```bash
# 1. Start from clean main
git checkout main && git pull

# 2. Bump version (edit the single source of truth)
#    pyproject.toml -> version = "0.x.y"
vim pyproject.toml

# 3. Update CHANGELOG.md
#    Move items from [Unreleased] to [0.x.y] - YYYY-MM-DD
vim CHANGELOG.md

# 4. Commit the release
git add pyproject.toml CHANGELOG.md
git commit -m "release: v0.x.y"

# 5. Tag and push (triggers CI -> PyPI + GitHub release)
git tag v0.x.y
git push && git push --tags

# 6. Verify (wait ~2 min for CI)
gh run watch                              # watch the release workflow
pip install vllm-factory==0.x.y --no-cache-dir  # verify PyPI
```

## Hotfix workflow

For urgent fixes to a released version:

```bash
# 1. Branch from the release tag
git checkout -b hotfix/fix-name v0.x.y

# 2. Fix, commit, push
git push -u origin HEAD && gh pr create --base main

# 3. After merge to main, tag the hotfix release
git checkout main && git pull
# Bump patch version in pyproject.toml (0.x.y+1)
git add pyproject.toml CHANGELOG.md
git commit -m "release: v0.x.y+1"
git tag v0.x.y+1 && git push && git push --tags
```

## PyPI trusted publisher setup (one-time)

This is already configured in `.github/workflows/release.yml` using OIDC.
You only need to do this once:

1. Go to [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/)
2. Click "Add a new pending publisher"
3. Fill in:
   - PyPI project name: `vllm-factory`
   - Owner: `latenceainew`
   - Repository: `vllm-factory`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
4. Go to GitHub repo Settings -> Environments -> Create environment named `pypi`
5. Push a tag to test: `git tag v0.0.0-test && git push origin v0.0.0-test`
6. Delete the test tag after: `git push origin :v0.0.0-test && git tag -d v0.0.0-test`

### Fallback: API token method

If you prefer not to use OIDC trusted publishers:

1. Create an API token at [pypi.org/manage/account/#api-tokens](https://pypi.org/manage/account/#api-tokens)
2. Add it as a GitHub secret: Settings -> Secrets -> `PYPI_API_TOKEN`
3. Update `.github/workflows/release.yml` pypi-publish step:
   ```yaml
   - uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

## Parity testing before release

Always run the full parity suite before tagging a release:

```bash
make test-all    # runs all 12 plugins via vllm serve
```

All 12 must pass. If any fail, do not release.

## Useful commands

| Task | Command |
|---|---|
| Run all parity tests | `make test-all` |
| Test single plugin | `make test-serve P=colqwen3` |
| Lint | `make lint` |
| Full CI locally | `make ci-test` |
| Serve a plugin | `make serve P=moderncolbert` |
| Check current version | `python -c "import importlib.metadata; print(importlib.metadata.version('vllm-factory'))"` |
| List open PRs | `gh pr list` |
| Create PR | `gh pr create` |
| Watch CI | `gh run watch` |
