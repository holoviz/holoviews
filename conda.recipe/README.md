## Release Procedure

- Ensure all tests pass.

- Tag commit a PEP440 style tag (starting with the prefix 'v') and push to github

```bash
git tag -a vx.x.x -m 'Version x.x.x'
git push --tags
```

Example tags might include v1.9.3 v1.10.0a1 or v1.11.3b3

- Build conda packages
