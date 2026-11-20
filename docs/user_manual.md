# User manual

This guide is for **students** using the assistant to study, and for **course staff** who manage the
materials. Your trainer will give you the address to open — it ends in `:8080` or similar.

## Signing in

Choose the person icon at the bottom of the sidebar and sign in with your Google account. The first
time, you'll be asked for the name you'd like to be called. You can use Ask and all the practice screens
without signing in, but only signed-in students have their conversations and scores saved.

If the page says *"Signing in will not stick on this address"*, use the link it shows — the address must
match exactly for sign-in to work.

## For students

### Ask
Type a question about your course and press **Enter** (Shift+Enter starts a new line).

- The answer appears word by word. **Sources** for the answer are listed on the right (or under the
  answer on a small screen), with the document and page — that is where to read more.
- An answer with a **green edge** comes from your course materials. A **grey edge** means the materials
  didn't cover your question and the answer is general knowledge — check it with your trainer before
  relying on it.
- Ask follow-ups naturally: *"what about the second one?"* or *"why is it faster?"* — the assistant
  remembers the conversation.
- **All subjects** (next to the Send button) narrows the search to one subject when your question could
  match several.
- **Checked answers** checks each answer against your materials before showing it. It is slower and
  the answer appears all at once rather than word by word. If an answer can't be verified, it is sent to
  a member of staff and you'll see *Waiting for staff review*: choose **Check again** later to see the
  reviewed answer.
- **New conversation** starts fresh. Rate any answer with the thumbs up or down — it helps staff improve
  the materials.

### My course
All the documents you can study, grouped by subject. Each card shows your best quiz score, how many
flashcards you know, and your interview rating (when signed in), with buttons to open its guide, quiz,
flashcards or interview practice.

### Guide
A one-page overview of a document: what it covers, its main points and the key terms to know. The first
time a guide is opened it takes 10–20 seconds to prepare; after that it opens instantly. Use **Take a
quiz** or **Flashcards** at the bottom to check what stuck.

### Quiz
Pick a document and choose **Start quiz**. Five multiple-choice questions are written from different
parts of it (20–40 seconds). After each answer you'll see whether you were right, why, and which page to
read again. Your first answer to each question is the one that counts.

### Flashcards
Pick a document and choose **Make flashcards**. Try to answer each card, then click it to flip. Choose
**I knew it** or **Still learning** honestly — at the end you can go over just the cards you missed.

### Placement
Practice for job interviews. Pick a document and choose **Start practice**: you'll get a question an
interviewer could ask about what that material taught you. Write your answer the way you'd say it, then
**Get feedback**. You'll see a rating out of 5, which of the key points you covered, partly covered or
missed, what went well, what to add, and an example of a strong answer.

### Progress
Your questions asked, quiz average, flashcards known, interview average, activity over the last 14 days,
and results by document. Every number is counted from what you actually did. Sign in to see it.

## For course staff

Staff are set up by the system administrator (in the `ADMIN_EMAILS` setting). Signed in as staff you
see two extra things: the controls on **Documents**, and the **Monitoring** screen.

### Documents
- **Add a handout**: drag a file onto the box, or choose **Choose a file**. PDF, Word, Markdown, text
  and CSV are supported. A scanned PDF with no text layer can't be read — the screen will say so.
- **Add a web page**: paste its address and choose **Add page**. Pages that build their content with
  JavaScript may have no readable text.
- **Remove**: the bin icon on a row. Students immediately stop getting answers from that document; the
  uploaded file itself is kept.
- Uploading a new version of a file with the same name replaces the old version.

### Monitoring
- **Requests served, answer time, satisfaction** — how the assistant is being used and rated.
- **Human review queue** — answers the checked workflow could not verify against the materials. Read
  the question, the draft and the statements that couldn't be supported. **Approve** releases the
  answer (edit the draft first if needed); **Reject** tells the student to ask their trainer.
- **Evaluation** — the latest results of the automated test set: how often answers cite the right
  document, stay faithful to it, admit when the materials don't cover something, and resist attempts
  to misuse the assistant.
- **Controlled workflow** — a diagram of the steps every checked answer goes through.

Choose **Refresh** to load the latest numbers.

## When something goes wrong

| You see | What to do |
|---|---|
| "Could not reach the assistant" | The service may be restarting — wait a minute and try again |
| An answer seems wrong | Check the cited page; rate it thumbs down so staff can review it |
| A document is missing | Ask your trainer to add it under Documents |
| "Only staff can manage the knowledge base" | You need a staff account to add or remove documents |
