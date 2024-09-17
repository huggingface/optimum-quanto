<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribute to optimum-quanto

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md).

**This guide is directly inspired by [transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Implement new kernels.

> All contributions are equally valuable to the community. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](https://github.com/huggingface/optimum-quanto/blob/main/CONTRIBUTING.md/#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The `optimum-quanto` backend will become more robust and reliable thanks to users who will report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code. If you're unsure whether the bug is in your code or the library, please ask in the [forum](https://discuss.huggingface.co/) first. This helps us respond quicker to fixing issues related to the library versus general questions.

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so we can quickly resolve it:

* Your **OS type and version** and **Python** and **PyTorch** versions.
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.

### Do you want a new feature?

If there is a new feature you'd like to see, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it a feature related to something you need for a project? Is it something you worked on and think it could benefit the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the features usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

## Do you want to implement a new kernel?

With the constant evolution of hardware backends, there is always a need for updating the kernels for better performance.

* The hardware configuration(s) it will apply to.
* If any, a short description of the novel techniques that should be used to implement the kernel.

If you are willing to contribute the kernel yourself, let us know so we can help you add it to `optimum-quanto`!

## Create a Pull Request

Before writing any code, we strongly advise you to search through the existing PRs or
issues to make sure nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to contribute. While `git` is not the easiest tool to use, it has the greatest manual. Type `git --help` in a shell and enjoy! If you prefer books, [Pro Git](https://git-scm.com/book/en/v2) is a very good reference.

You'll need **Python 3.8** or above to contribute. Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/huggingface/optimum-quanto) by
   clicking on the **[Fork](https://github.com/huggingface/optimum-quanto/fork)** button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/optimum-quanto.git
   cd optimum-quanto
   git remote add upstream https://github.com/huggingface/optimum-quanto.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   pip install -e ".[dev]"
   ```

   If `optimum-quanto` was already installed in the virtual environment, remove
   it with `pip uninstall optimum-quanto` before reinstalling it in editable
   mode with the `-e` flag.

5. Develop the features in your branch.

   As you work on your code, you should make sure the test suite
   passes. Run the tests impacted by your changes like this:

   ```bash
   pytest test/<TEST_TO_RUN>.py
   ```

   `optimum-quanto` relies on `black` and `ruff` to format its source code
   consistently. After you make changes, apply automatic style corrections and code verifications
   that can't be automated in one go with:

   ```bash
   make style
   ```
   Once you're happy with your changes, add the changed files with `git add` and
   record your changes locally with `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   This repository uses a `rebase` strategy when merging pull-requests, meaning that your commits will **not** be squashed automatically.

   We therefore request you to keep a tidy queue of commits in your pull-request, clearly communicating the changes you made in each commit.

   **This is enforced by the continuous integration, so your pull-request will not be reviewed if your commit queue is not clean.**

   Although this is not mandatory, we kindly ask you to consider using [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)
   (here the full [specification](https://www.conventionalcommits.org/en/v1.0.0/))!

   This article gives a brief [rationale](https://julien.ponge.org/blog/the-power-of-conventional-commits/) of why this will make our life and yours easier.

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Before submitting, cleanup your commit history to make it more readable for the reviewer (like squashing temporary commits and editing commit messages to clearly explain what you changed).

   Push your changes to your branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

6. Now you can go to your fork of the repository on GitHub and click on **Pull Request** to open a pull request. Make sure you tick off all the boxes on our [checklist](https://github.com/huggingface/optimum-quanto/blob/main/CONTRIBUTING.md/#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review.

7. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Pull request checklist

☐ The pull request title should summarize your contribution.<br>
☐ If your pull request addresses an issue, please mention the issue number in the pull
request description to make sure they are linked (and people viewing the issue know you
are working on it).<br>
☐ To indicate a work in progress please prefix the title with `[WIP]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.<br>
☐ Make sure existing tests pass.<br>
☐ If adding a new feature, also add tests for it.<br>
☐ All public methods must have informative docstrings.<br>

### Tests

An extensive test suite is included to test the library behavior in the [test](https://github.com/huggingface/optimum-quanto/tree/main/test) folder.

From the root of the repository, specify a *path to a subfolder or a test file* to run the test.

```bash
python -m pytest -sv ./test/<subfolder>/<test>.py
```

You can run all tests by typing:

```bash
make test
```

### Style guide

For documentation strings, `optimum-quanto` follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Check `transformers` [documentation writing guide](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification)
for more information.
