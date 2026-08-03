# Getting started

**No coding needed. No GitHub account needed.** If you can copy and paste one line, you can
run this.

About 10 minutes, most of it downloading.

---

## What you need

| | |
|---|---|
| An **NVIDIA graphics card** | The kind used for gaming. If your PC has one, you are fine. |
| **Windows 10/11 or Linux** | No Mac version — Macs do not have NVIDIA cards. |
| **About 8 GB of free space** | The AI model itself is around 4 GB. |

You do **not** need Python, CUDA, PyTorch, a GitHub account, or any programming knowledge.
The installer handles all of it.

---

## Step 1 — Open a terminal

A "terminal" is a window where you type commands. It looks intimidating and isn't.

<details>
<summary><b>On Windows</b> — click to expand</summary>

You need Ubuntu, which runs inside Windows. One-time setup:

1. Click **Start**, type `PowerShell`, right-click it, choose **Run as administrator**.
2. Type this and press Enter:

   ```
   wsl --install
   ```

3. **Restart your computer** when it asks.
4. After restarting, click **Start**, type `Ubuntu`, and open it.
5. The first time, it asks you to make a username and password. Pick anything — the
   password will not show as you type, which is normal.

That Ubuntu window is your terminal. Use it for Step 2.
</details>

<details>
<summary><b>On Linux</b> — click to expand</summary>

Press **Ctrl + Alt + T**. That's it.
</details>

---

## Step 2 — Paste one line

Copy this, paste it into the terminal, and press Enter:

```bash
curl -fsSL https://raw.githubusercontent.com/Questeria/HIVE-Limited/main/install.sh | bash
```

> **Pasting into a terminal:** `Ctrl + V` often does nothing there. Use **right-click**, or
> **Ctrl + Shift + V**.

The installer checks your computer, downloads everything, and tells you what it's doing. If
something is missing it says exactly what to type to fix it.

Leave it running. The model download is a few GB and takes the longest.

---

## Step 3 — Open it

When it finishes it offers to start. Say yes, then open your web browser to:

**http://localhost:8080**

Type a question, press **LAUNCH**, and watch it answer.

To start it again another day:

```bash
~/hive-limited/start.sh
```

---

## Racing it against another AI engine

The whole point of the arena is that you don't take our word for anything — you run the
other engine yourself and watch them side by side.

Click **➕ ADD AN ENGINE** at the top of the page, or run:

```bash
~/hive-limited/setup/install_llamacpp.sh
```

That installs **llama.cpp**, the most popular way to run AI models locally. Then reload the
page, pick it from the dropdown, and press LAUNCH. Both engines answer the same question and
the page shows what it just measured.

Every number you see was produced by the run you just watched, on your machine. Nothing is
pre-recorded.

---

## If something goes wrong

**First, run the doctor.** It checks everything and prints the exact fix for anything wrong:

```bash
~/hive-limited/doctor.sh
```

**"curl: command not found"**
```bash
sudo apt update && sudo apt install -y curl
```
then paste the Step 2 line again.

**"No NVIDIA graphics driver found"**
On Windows, install the normal driver from [nvidia.com/drivers](https://www.nvidia.com/drivers)
— in *Windows*, not inside Ubuntu — then close and reopen the Ubuntu window. You do **not**
need the big "CUDA Toolkit".

**The download stopped partway**
Run the same line again. It keeps what it already downloaded and carries on.

**"Permission denied"**
```bash
chmod +x ~/hive-limited/start.sh
```

**The page won't open in the browser**
Check the terminal still shows it running. If you closed that window, the program stopped —
start it again with `~/hive-limited/start.sh`.

**Something else**
Email **ajdemarco10@gmail.com** with what you typed and what it said. A copy-paste of the
error is enough.

---

## What this is

A deliberately **limited** demonstration build. It answers one question at a time and is not
the full HIVE engine — the fast paths, the compiler and the research record are not included
here. Everything it shows you, it measured on your machine while you watched.

Results from the full engine, on the hardware they were measured on, are in
[REFERENCE.md](REFERENCE.md) — including the places where NVIDIA is still ahead of us.

To see the unrestricted engine, email **ajdemarco10@gmail.com**.
