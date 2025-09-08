# Hướng dẫn chạy 

# 1. Local 
- cd src/ChatApp
- mvn clean compile
- mvn javafx:run

# 2. Docker
- cd src/ChatApp
- docker build -t chatapp:latest .
-  docker run --rm -p 5900:5900 -p 6080:6080 chatapp:latest
- Mở trình duyệt và truy cập http://localhost:6080
- Click chọn vnc.html
- Click chọn Connect
