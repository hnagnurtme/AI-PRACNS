#!/usr/bin/env node
import { GoogleGenerativeAI } from "@google/generative-ai";
import { execSync } from "child_process";
import readline from "readline";

async function main() {
    const apiKey = process.env.GOOGLE_API_KEY;
    if ( !apiKey ) {
        console.error( "❌ Missing GOOGLE_API_KEY. Run:" );
        console.error( '   export GOOGLE_API_KEY="YOUR_KEY_HERE"' );
        process.exit( 1 );
    }

    const genAI = new GoogleGenerativeAI( apiKey );
    const model = genAI.getGenerativeModel( { model: "gemini-1.5-flash-latest" } );

    let diff = "";
    try {
        diff = execSync( "git diff --cached", { encoding: "utf8" } );
        if ( !diff.trim() ) {
            console.error( "⚠️ No staged changes found. Run `git add .` first." );
            process.exit( 1 );
        }
    } catch ( e ) {
        console.error( "❌ Failed to get git diff:", e.message );
        process.exit( 1 );
    }

    console.log( "🤖 Generating commit message using Gemini...\n" );

    const prompt = `
You are an assistant that generates **Git commit messages** following the **Conventional Commits** format.

---
### Rules:
- Use lowercase type followed by a colon and a short title.  
- Keep the title under 72 characters.
- Do not include trailing punctuation.
- Use **English**, concise, and clear.
- Include a scope if relevant (e.g., "client", "api", "db").
- Focus on purpose and outcome, not implementation details.

### Allowed types:
- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Code style (formatting, missing semicolons, etc)
- refactor: Code change that neither fixes a bug nor adds a feature
- perf: Performance improvement
- test: Adding or modifying tests
- chore: Build process, dependency update, or other non-code changes

### Examples:
- feat(api): add retry mechanism for failed requests
- fix(client): resolve packet duplication issue
- refactor(core): simplify state handling logic
- chore(deps): update MongoDB driver to latest version

Now analyze this staged git diff and generate ONE appropriate Conventional Commit message:

${ diff }
`;

    try {
        const result = await model.generateContent( prompt );
        const commitMsg = result.response.text().trim();

        console.log( "✅ Suggested commit message:\n" );
        console.log( `   ${ commitMsg }\n` );

        const rl = readline.createInterface( {
            input: process.stdin,
            output: process.stdout
        } );

        rl.question( "💬 Do you want to commit with this message? (Y/n): ", answer => {
            rl.close();
            if ( answer.trim().toLowerCase() === "y" || answer.trim() === "" ) {
                try {
                    execSync( `git commit -m "${ commitMsg.replace( /"/g, '\\"' ) }"`, { stdio: "inherit" } );
                    console.log( "\n🚀 Commit created successfully!" );
                } catch ( e ) {
                    console.error( "❌ Failed to run git commit:", e.message );
                }
            } else {
                console.log( "🛑 Commit canceled." );
            }
        } );
    } catch ( e ) {
        console.error( "❌ Gemini API error:", e.message );
    }
}

main();
