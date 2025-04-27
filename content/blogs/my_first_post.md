---
title: "My First Post"
date: 2022-11-11T09:50:56+01:00
author: Ali S.
draft: false
---

While I was building this website, I needed to write some content to test rendering and some of Hugo features. One of the good ideas I had was to write down how to build it and that can help people who liked my website and wanted to build something similar (or completely different, since the procedure is similar for most hugo projects).

Tools, like Hugo and Gatsby, make static website development fast and easy, especially thanks to the availability of templates, projects developed by numerous members of their respective communities. I hope this serve for the same purpose.

First, install hugo on your machine using `apt-get` or `dnf` on linux, `brew` on mac and `choco` on windows.

Then, if you're a n00b like me, just run:

```bash
hugo new site my-hugo-website
```

This will create a folder named `my-hugo-website/` that contains the basic stuff (a bunch of folders and a `config.toml` file) you will need in a Hugo website project.

The project tree would be something like this:

```bash
$ tree ./
.
├── archetypes
│ └── default.md
├── config.toml
├── content
├── data
├── layouts
├── public
├── resources
│ └── _gen
│ ├── assets
│ └── images
├── static
└── themes
```

Then, you can import a Hugo theme/template that you like and suits your project. Many are available on github (see [gohugoio/HugoThemes](https://github.com/gohugoio/hugoThemes)) or some template galleries like [Jamstack themes](https://jamstackthemes.dev/) where I found the template I am using for this project ([hugo-profile](https://github.com/gurusabarish/hugo-profile)).

To import the theme with git, run:

```bash
git clone https://github.com/gurusabarish/hugo-profile.git ./themes/hugo-profile
```

Once the theme is imported, you can start configuring the `config.yaml` (or `config.toml`) file to tell hugo that you want to use that specific theme.

Actually, the `config.toml` allows many options to customize your site. For this template, I'm using YAML format instead:

```yaml
baseURL: "https://alisaghiran.github.io"
languageCode: "en-us"
title: "Ali's Personal Page"
theme: hugo-profile

# Theme specific parameters
params:
  title: "Ali's Personal Page"
  description: "Personal website and blog"
  
  # Profile settings
  profile:
    enable: true
    name: "Ali S."
    tagline: "Developer & Researcher"
    image: "images/profile.jpg"
    
  # Navigation menu
  menu:
    enable: true
    items:
      - name: "Home"
        url: "#"
        weight: 1
      - name: "Blog"
        url: "/blog"
        weight: 2
      - name: "About"
        url: "/about"
        weight: 3
```

To create your first post (like this one), use the command:

```bash
hugo new posts/my-first-post.md
```

This creates a markdown file with front matter (the section between --- marks at the top).

To preview your site locally, run:

```bash
hugo server -D
```

And visit <http://localhost:1313> in your browser.

When you're ready to deploy, run:

```bash
hugo
```

This generates your static site in the `public/` directory, which you can then deploy to GitHub Pages or any web hosting service.

That's it! You now have a fully functional Hugo website.
