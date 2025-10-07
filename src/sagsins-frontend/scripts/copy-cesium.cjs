#!/usr/bin/env node
const fs = require('fs')
const path = require('path')

const root = path.resolve(__dirname, '..')
const srcBase = path.join(root, 'node_modules', 'cesium', 'Build', 'Cesium')
const destBase = path.join(root, 'public', 'Cesium')

const folders = ['Assets', 'Workers', 'ThirdParty', 'Widgets']

function ensureDir(p) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true })
}

async function copyRecursive(src, dest) {
  if (fs.cp) {
    return fs.promises.cp(src, dest, { recursive: true })
  }

  ensureDir(dest)
  const entries = await fs.promises.readdir(src, { withFileTypes: true })
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name)
    const destPath = path.join(dest, entry.name)
    if (entry.isDirectory()) {
      await copyRecursive(srcPath, destPath)
    } else if (entry.isSymbolicLink()) {
      const real = await fs.promises.readlink(srcPath)
      await fs.promises.symlink(real, destPath)
    } else {
      await fs.promises.copyFile(srcPath, destPath)
    }
  }
}

async function main() {
  ensureDir(destBase)

  for (const folder of folders) {
    const src = path.join(srcBase, folder)
    const dest = path.join(destBase, folder)
    if (!fs.existsSync(src)) {
      console.warn(`Skipping missing Cesium folder: ${src}`)
      continue
    }
    console.log(`Copying ${src} -> ${dest}`)
    try {
      await copyRecursive(src, dest)
      console.log(`Copied ${folder} successfully`)
    } catch (err) {
      console.error(`Failed to copy ${folder}:`, err)
      process.exitCode = 2
    }
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
