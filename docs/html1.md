# HTML Basics

## What is HTML?

HTML (HyperText Markup Language) is the standard markup language used to create web pages. It provides the structure of a webpage by organizing content using different elements called tags.

A browser reads HTML documents and displays the content to users.

---

## Basic Structure of an HTML Document

Every HTML document follows a standard structure.

```html
<!DOCTYPE html>
<html>
<head>
    <title>My First Page</title>
</head>
<body>

<h1>Hello World</h1>

</body>
</html>
```

### Explanation

- `<!DOCTYPE html>` tells the browser that this is an HTML5 document.
- `<html>` is the root element.
- `<head>` contains metadata about the webpage.
- `<title>` sets the page title.
- `<body>` contains everything visible to the user.

---

## HTML Elements

An HTML element usually consists of:

- Opening tag
- Content
- Closing tag

Example:

```html
<p>This is a paragraph.</p>
```

Some elements are self-closing.

Example:

```html
<img src="image.png" alt="Image">
```

---

## Headings

HTML provides six heading levels.

```html
<h1>Main Heading</h1>
<h2>Sub Heading</h2>
<h3>Smaller Heading</h3>
<h4>Heading Four</h4>
<h5>Heading Five</h5>
<h6>Heading Six</h6>
```

`<h1>` is the most important heading.

---

## Paragraphs

Paragraphs are created using the `<p>` tag.

```html
<p>This is my first paragraph.</p>
```

---

## Links

Links allow navigation between pages.

Example:

```html
<a href="https://example.com">Visit Example</a>
```

The `href` attribute specifies the destination URL.

---

## Images

Images are inserted using the `<img>` tag.

Example:

```html
<img src="cat.jpg" alt="Cute Cat">
```

Important attributes:

- src
- alt
- width
- height

---

## Lists

### Ordered List

```html
<ol>
    <li>Wake up</li>
    <li>Study</li>
    <li>Sleep</li>
</ol>
```

### Unordered List

```html
<ul>
    <li>Apple</li>
    <li>Mango</li>
    <li>Orange</li>
</ul>
```

---

## Tables

Tables organize data into rows and columns.

Example:

```html
<table>
<tr>
<th>Name</th>
<th>Age</th>
</tr>

<tr>
<td>Alice</td>
<td>21</td>
</tr>
</table>
```

Main tags:

- table
- tr
- th
- td

---

## Forms

Forms collect user input.

Example:

```html
<form>

<label>Name</label>

<input type="text">

<input type="submit">

</form>
```

Common input types:

- text
- email
- password
- checkbox
- radio
- number
- date

---

## Semantic HTML

Semantic HTML improves readability and accessibility.

Examples:

- header
- nav
- main
- article
- section
- aside
- footer

Semantic elements help search engines understand webpage structure.

---

## Best Practices

1. Use semantic HTML whenever possible.
2. Keep code properly indented.
3. Write meaningful alt text for images.
4. Avoid unnecessary nested elements.
5. Use headings in the correct order.
6. Validate HTML before deployment.

---

## Summary

HTML is the foundation of every website. It defines the structure of webpages using elements and tags. Common HTML components include headings, paragraphs, links, images, lists, tables, and forms. Semantic HTML makes websites easier to understand for both users and search engines.