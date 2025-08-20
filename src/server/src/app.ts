import express, { Application, Request, Response } from "express";

const app: Application = express();

// Middleware
app.use(express.json());

// Route mẫu
app.get("/", (req: Request, res: Response) => {
  res.send("Server is running 🚀");
});

export default app;