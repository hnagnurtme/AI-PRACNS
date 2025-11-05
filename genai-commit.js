import 'dotenv/config'; // ✅ Tự động load GOOGLE_API_KEY từ .env
import { GoogleGenerativeAI } from "@google/generative-ai";
import { execSync } from "child_process";
import readline from "readline";

async function main() {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) {
        console.error("❌ Missing GOOGLE_API_KEY in .env file");
        console.error("👉 Please create a .env file with:");
        console.error('   GOOGLE_API_KEY="YOUR_KEY_HERE"');
        process.exit(1);
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" })

    let diff = "";
    try {
        diff = execSync("git diff --cached", { encoding: "utf8" });
        if (!diff.trim()) {
            console.error("⚠️ No staged changes found. Run `git add .` first.");
            process.exit(1);
        }
    } catch (e) {
        console.error("❌ Failed to get git diff:", e.message);
        process.exit(1);
    }

    console.log("🤖 Generating commit message using Gemini...\n");

    const prompt = `
You are an assistant that generates **Git commit messages** following the **Conventional Commits** standard.

### Rules:
- Format: \`<type>(<optional scope>): <short summary>\`
- Keep summary under 72 characters.
- Be clear and concise — describe **what** and **why**, not **how**.
- Avoid punctuation at the end.
- Use active verbs (add, fix, improve, remove, refactor, optimize).
- Detect scope from diff path (client, server, rl, infra, etc.)

### Allowed types:
feat, fix, docs, style, refactor, perf, test, chore

### Examples:
- feat(client): add UI for satellite visualization
- fix(rl): correct reward normalization issue
- chore(infra): update Docker Compose setup

Now analyze this git diff and generate **one valid Conventional Commit message**:

${diff}
`;

    try {
        const result = await model.generateContent(prompt);
        const commitMsg = result.response.text().trim().split("\n")[0]; // Lấy dòng đầu tiên

        console.log("✅ Suggested commit message:\n");
        console.log(`   ${commitMsg}\n`);

        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        rl.question("💬 Do you want to commit with this message? (Y/n): ", (answer) => {
            rl.close();
            if (answer.trim().toLowerCase() === "y" || answer.trim() === "") {
                try {
                    execSync(`git commit -m "${commitMsg.replace(/"/g, '\\"')}"`, { stdio: "inherit" });
                    console.log("\n🚀 Commit created successfully!");
                } catch (e) {
                    console.error("❌ Failed to run git commit:", e.message);
                }
            } else {
                console.log("🛑 Commit canceled.");
            }
        });
    } catch (e) {
        console.error("❌ Gemini API error:", e.message);
    }
}

main();
