# Hướng dẫn chạy 

# 1. Local 
- cd src/ChatApp
- mvn clean compile
- mvn javafx:run

# 2. Docker
- cd src/ChatApp
- docker build -t chatapp:latest .
docker network create chat-network

# 3. Start server
docker run -d --name chat-server --network chat-network -p 8888:8888 -e SERVER_ONLY=true chatapp

# 4. Start client 1  
docker run -d --name chat-user1 --network chat-network -p 6080:6080 -e SOCKET_SERVER_HOST=chat-server chatapp

# 5. Start client 2
docker run -d --name chat-user2 --network chat-network -p 6081:6080 -e SOCKET_SERVER_HOST=chat-server chatapp

- Click chọn vnc.html
- Click chọn Connect
