## Pull Request (PR) Checklist

**Before submitting, please ensure your PR title follows the conventional commit format:**

```
<type>([scope]): <subject>
```

### Valid Types:

- `build:` - Changes to build system or dependencies
- `chore:` - Routine tasks, maintenance
- `ci:` - CI/CD configuration changes
- `compat:` - Compatibility updates
- `docs:` - Documentation changes
- `enh:` - New features or enhancements
- `feat:` - Enhancements
- `fix:` - Bug fixes
- `perf:` - Performance improvements
- `refactor:` - Code refactoring
- `test:` - Test additions or modifications
- `type:` - Type annotation changes

### Valid Scopes (optional):

- `dev` - Development tooling
- `data` - Data handling
- `plotting` - General plotting
- `bokeh` - Bokeh backend
- `matplotlib` - Matplotlib backend
- `plotly` - Plotly backend
- `notebook` - Notebook integration

### Examples:

- ✅ `fix(bokeh): correct hover tooltip positioning`
- ✅ `docs: update installation guide`
- ✅ `feat(plotting): add new colormap options`
- ❌ `Fix hover tooltip` (missing type and colon)
- ❌ `Fix: hover tooltip` (subject shouldn't start with uppercase)

---

## Description

<!-- Please describe your changes here -->

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Other

## Checklist

- [ ] My PR title follows the conventional commit format shown above
- [ ] I have added tests
- [ ] I have updated the documentation (if applicable)
