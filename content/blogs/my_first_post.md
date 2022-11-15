---
title: "My First Post"
date: 2022-11-11T09:50:56+01:00
author: Ali S.
draft: true
---

*Or how did I build this website?*

While I was building this website, I needed to write some content to test rendering and some of Hugo features. One of the good ideas I had was to write down how to build it and that can help people who liked my website and wanted to build something similar (or completely different, since the procedure is similar for most hugo projects).
Tools, like Hugo and Gatsby, make static website development fast and easy, especially thanks to the availability of templates, projects developed by numerous members of their respective communities. I hope this serve for the same purpose.

First, install hugo on your machine using `apt-get` or `dnf` on linux, `brew` on mac and `choco` on windows.
Then, if you're a n00b like me, just run:

```{bash}
hugo new site my-hugo-website
```

This will create a folder named `my-hugo-website/` that contains the basic stuff (a bunch of folders and a `config.toml` file) you will need in a Hugo website project.
The project tree would be something liek this :

```{bash}
$ tree ./
.
├── archetypes
│   └── default.md
├── config.toml
├── content
├── data
├── layouts
├── public
├── resources
│   └── _gen
│       ├── assets
│       └── images
├── static
└── themes
```

Then, you can import a Hugo theme/template that you like and suits your project. Many are available on github (see [gohugoio/HugoThemes](https://github.com/gohugoio/hugoThemes)) or some template galleries like [Jamstack themes](https://jamstackthemes.dev/) where i found the template I am using for this project ([hugo-profile](https://github.com/gurusabarish/hugo-profile)).

To import the theme with git, run:

```{bash}
git clone https://github.com/gurusabarish/hugo-profile.git ./themes/hugo-profile
```

TBC.