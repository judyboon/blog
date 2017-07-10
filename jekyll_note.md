# Tips for generating website with equations in Github Pages using jekyll

## Install jekyll

## Basic steps:

1. Run `jekyll new website_name` to create a website project

2. Go into 'website_name' folder

3. Make sure the jekyll version used next is consistent with its
   github version

   1. comment `gem "jekyll", "3.5.0"`
   2. uncomment `gem "github-pages", group: :jekyll_plugins`
   3. run `bundle update; bundle install`
   4. Check [link](https://github.com/github/pages-gem) for running a
      local version of jekyll which is consistent with Github Pages
      version

4. Push local folder to a git repo


## Add equations to markdown

1. Run `bundle show minima` to locate the directory of theme html
   folders

2. Locate '_includes/' folder. If it is not in jekyll website
   directory, create one in the directory.

3. add javascript into '_includes/head.html' in website
   directory. Check
   [link](http://csega.github.io/mypost/2017/03/28/how-to-set-up-mathjax-on-jekyll-and-github-properly.html)


## Create site locally using jekyll


## Tips

- Use `bundle exec jekyll new site_name` when `jekyll new site_name`
is not working


