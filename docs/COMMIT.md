# 📝 Commit Message Guidelines

## 🎯 Format Chuẩn

```
<type>(<scope>): <subject> #<issue-number>

[optional body]

[optional footer]
```

### Với Issue Number:
```
feat(auth): add login API #12
fix(api): resolve null pointer exception #34
docs(readme): update setup instructions #56
```

## 📦 Types

| Type | Mô tả | Ví dụ |
|------|-------|-------|
| `feat` | Thêm tính năng mới | `feat(auth): add login API #12` |
| `fix` | Sửa bug | `fix(api): resolve null pointer exception #34` |
| `docs` | Cập nhật documentation | `docs(readme): update setup instructions #56` |
| `style` | Format code (không ảnh hưởng logic) | `style(login): format indentation #78` |
| `refactor` | Refactor code (không thêm feature/fix bug) | `refactor(auth): simplify token validation #90` |
| `perf` | Cải thiện performance | `perf(query): optimize database query #45` |
| `test` | Thêm/sửa tests | `test(auth): add unit tests for login #67` |
| `chore` | Maintenance tasks | `chore(deps): update dependencies #89` |
| `build` | Build system changes | `build(docker): update Dockerfile #23` |
| `ci` | CI/CD changes | `ci(github): add deployment workflow #11` |
| `revert` | Revert commit trước | `revert: feat(auth): add login API #12` |

## 🎨 Scope (Optional)

Scope là module/component bị ảnh hưởng:

**Backend:**
- `auth` - Authentication
- `api` - API endpoints
- `db` - Database
- `service` - Business logic services
- `middleware` - Middleware

**Frontend:**
- `ui` - UI components
- `page` - Pages/routes
- `hook` - Custom hooks
- `store` - State management
- `api` - API integration

**Shared:**
- `config` - Configuration
- `deps` - Dependencies
- `test` - Testing

## ✍️ Subject Rules

1. ✅ Dùng imperative mood ("add" không phải "added" hay "adds")
2. ✅ Không viết hoa chữ cái đầu
3. ✅ Không có dấu chấm (.) ở cuối
4. ✅ Tối đa 50 ký tự
5. ✅ Mô tả ngắn gọn WHAT thay vì HOW

### ✅ Good Examples
```
feat(auth): add JWT token validation #42
fix(api): handle null user in login endpoint #15
docs(api): update API documentation #28
refactor(service): extract user validation logic #33
perf(db): add index to user email column #51
```

### ❌ Bad Examples
```
feat(auth): Added JWT token validation feature    # "Added" thay vì "add", thiếu issue number
fix(api): Fix bug. #12                            # Quá chung chung, có dấu chấm, viết hoa
Updated documentation                              # Không có type, không có issue number
refactor: changed some code in auth service #9    # "changed" thay vì "change"
auth: new login feature with JWT and OAuth        # Quá dài, không có type, thiếu issue number
feat(auth): add login #12345678                   # Issue number không hợp lệ
```

## 📄 Body (Optional)

- Dùng khi cần giải thích CHI TIẾT hơn
- Wrap ở 72 ký tự
- Giải thích **WHY** (tại sao) và **WHAT** (cái gì), không phải **HOW** (như thế nào)
- Cách subject 1 dòng trống

### Example:
```
feat(auth): add refresh token mechanism #87

Implement refresh token to improve security and user experience.
Access tokens now expire after 15 minutes, refresh tokens after 7 days.
This prevents long-lived tokens from being compromised.
```

## 🔗 Footer (Optional)

Dùng để reference issues hoặc breaking changes:

### Reference Issues
```
feat(api): add user profile endpoint #123

Closes #123
Related to #456
```

### Breaking Changes
```
feat(api): change authentication response format #789

BREAKING CHANGE: The API now returns { success, data, error } 
instead of direct data object. Update all API calls accordingly.

Closes #789
```

## 🚀 Quick Reference

### Frontend Examples
```bash
# Thêm component mới
git commit -m "feat(ui): add UserProfile component #42"

# Sửa bug UI
git commit -m "fix(page): resolve layout issue on mobile #15"

# Integrate API
git commit -m "feat(api): integrate login endpoint #28"

# Update styling
git commit -m "style(ui): update button colors to match design #33"
```

### Backend Examples
```bash
# Thêm API endpoint
git commit -m "feat(api): add POST /api/auth/login endpoint #12"

# Sửa bug logic
git commit -m "fix(service): handle empty email validation #24"

# Cập nhật database
git commit -m "feat(db): add User node schema to Neo4j #36"

# Performance improvement
git commit -m "perf(query): optimize graph traversal query #48"
```

### Common Examples
```bash
# Update dependencies
git commit -m "chore(deps): update React to v18.3.0 #51"

# Add tests
git commit -m "test(auth): add unit tests for login service #67"

# Update documentation
git commit -m "docs(readme): add setup instructions for Neo4j #72"

# CI/CD changes
git commit -m "ci(github): add automated testing workflow #89"
```

## 🎓 Tips cho Hackathon

1. **Commit nhỏ, commit thường** - Mỗi commit nên là 1 logical change
2. **Luôn thêm issue number** - Giúp track công việc và liên kết với task
3. **Commit trước khi chuyển task** - Đừng để code dang dở
4. **Push thường xuyên** - Tránh mất code khi có sự cố
5. **Review commit message trước khi push** - Dùng `git log` để kiểm tra

## 🔥 Workflow Example

```bash
# 1. Checkout branch từ issue
git checkout -b feat/login-api-12

# 2. Code và commit
git add .
git commit -m "feat(api): add login endpoint #12"

# 3. Thêm tests
git add .
git commit -m "test(api): add tests for login endpoint #12"

# 4. Update documentation
git add .
git commit -m "docs(api): document login endpoint #12"

# 5. Push
git push origin feat/login-api-12
```

## 📋 Issue Number Rules

- **Required**: Mọi commit PHẢI có issue number
- **Format**: `#<số>` (ví dụ: `#12`, `#456`)
- **Vị trí**: Cuối subject line, trước body
- **Multiple issues**: Dùng footer nếu liên quan nhiều issues

### Multiple Issues Example:
```
feat(api): add authentication system #12

Implement complete auth flow with login, logout, and token refresh.

Related to #13, #14
Closes #12
```

## 🎯 Quick Checklist

Trước khi commit, check:
- [ ] Type đúng? (feat, fix, docs, etc.)
- [ ] Scope rõ ràng? (auth, api, ui, etc.)
- [ ] Subject ngắn gọn? (< 50 chars)
- [ ] Dùng imperative mood? (add, fix, update)
- [ ] Không viết hoa chữ cái đầu?
- [ ] Không có dấu chấm cuối?
- [ ] **Có issue number?** (#12)
- [ ] Body giải thích đầy đủ? (nếu cần)

---

💡 **Pro tip**: Dùng Git alias để commit nhanh hơn:
```bash
# Add to ~/.gitconfig
[alias]
  cm = "!f() { git commit -m \"$1 #$2\"; }; f"
  
# Usage:
git cm "feat(api): add login endpoint" 12
```