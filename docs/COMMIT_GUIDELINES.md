# Quy tắc Commit

Dự án này tuân theo quy tắc [Conventional Commits](https://www.conventionalcommits.org/).

## Cấu trúc commit

`<type>(<scope>): <description> (#<issue-number>)`

### Các phần chi tiết
- **type**: loại thay đổi (bắt buộc)  
- **scope**: phạm vi thay đổi (tùy chọn, viết trong ngoặc)  
- **description**: mô tả ngắn gọn, rõ ràng, không viết hoa chữ cái đầu, không dấu chấm ở cuối  
- **issue-number**: số issue liên quan (tùy chọn, gắn bằng `#`)  

---

## Các loại commit (`type`)
- `feat`: Thêm tính năng mới  
- `fix`: Sửa lỗi  
- `docs`: Thay đổi tài liệu (README, wiki, …)  
- `style`: Thay đổi định dạng, code style (không ảnh hưởng logic)  
- `refactor`: Thay đổi mã nguồn nhưng không thêm tính năng, không sửa lỗi  
- `perf`: Cải thiện hiệu năng  
- `test`: Thêm hoặc sửa test  
- `build`: Thay đổi liên quan đến build system, dependency, package  
- `ci`: Thay đổi liên quan đến CI/CD  
- `chore`: Công việc lặt vặt, không ảnh hưởng đến src hoặc test  
- `revert`: Hoàn tác một commit trước đó  

---

## Ví dụ commit hợp lệ

### Thêm tính năng
`feat(auth): thêm chức năng đăng nhập bằng Google (#12)`

### Sửa lỗi
`fix(api): xử lý lỗi khi gọi API hết hạn token (#45)`

### Cập nhật tài liệu
`docs(readme): bổ sung hướng dẫn cài đặt (#10)`

### Refactor code
`refactor(user): tách lớp UserService ra riêng`

### Commit không liên quan issue
`chore: cập nhật version husky và commitlint`

---

## Gắn issue với commit

- Khi commit liên quan tới một issue cụ thể, **bắt buộc** thêm tag `#<issue-number>` ở cuối description.  
- Ví dụ:  
  - `feat(ui): thêm modal đăng ký (#23)`  
  - `fix(order): lỗi không lưu thông tin địa chỉ (#78)`  

---

## Lưu ý
- Mỗi commit chỉ nên mô tả **một thay đổi chính**.  
- Không commit file build (ví dụ: `dist/`, `node_modules/`, …).  
- Sử dụng tiếng Việt hoặc tiếng Anh thống nhất trong toàn bộ repo.  

---

## Quick Reference

| Type      | Mô tả                        | Ví dụ                                      |
|-----------|------------------------------|--------------------------------------------|
| `feat`    | Thêm tính năng mới           | `feat(auth): thêm đăng nhập Google (#12)`  |
| `fix`     | Sửa lỗi                      | `fix(api): sửa lỗi hết hạn token (#45)`    |
| `docs`    | Cập nhật tài liệu            | `docs(readme): thêm hướng dẫn cài đặt`     |
| `style`   | Thay đổi định dạng           | `style(ui): sửa indent component`          |
| `refactor`| Tái cấu trúc mã nguồn        | `refactor(user): tách UserService`         |
| `perf`    | Cải thiện hiệu năng          | `perf(db): tối ưu query tìm kiếm`          |
| `test`    | Thêm/sửa test                | `test(auth): thêm test đăng nhập`          |
| `build`   | Thay đổi build/dependency    | `build: cập nhật webpack version`          |
| `ci`      | Thay đổi CI/CD               | `ci: sửa pipeline deploy`                  |
| `chore`   | Công việc lặt vặt            | `chore: cập nhật husky`                    |
| `revert`  | Hoàn tác commit              | `revert: hoàn tác commit #123`           |