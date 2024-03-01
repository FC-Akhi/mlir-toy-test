# Some Useful Git commands

Why? Because I forget them everytime if I donot use them heavily. 😅

## 1. How to add git config (i.e. configure `.git/config`)

### 1.1. Add PROJECT-SPECIFIC `user.name` and `user.email`

- **How to set**

```sh
git config user.name "Your project specific name"
git config user.email "your@project-specific-email.com"
```

- **How to check/get**

```sh
git config --get user.name
git config --get user.email
```

### 1.2. Add Global `user.name` and `user.email`

One `$USERNAME` from OS will get that settings

- **How to set**

```sh
git config --global user.name "Your global username"
git config --global user.email "your@email.com"
```

- **How to check/get**

```sh
git config --global --get user.name
git config --global --get user.email
```


## 2. How to delete a local branch

- **To be safe, always try to find out the immediate mother/parent branch from where your target branch (i.e. the branch you want to delete) is originated**
- **Then switch to that immediate parent branch. And `--delete` it from there**

### 2.1. How to check

- **With command line**

```sh
git log --oneline --all --graph

# Returns
*   de8eadd (HEAD -> ch-0-2-prep-toy-scaffold, master) Merge branch 'ch-0-1-prep-mlir-template'
|\  
| * 2dfd228 (ch-0-1-prep-mlir-template) Minor changes 1. Docs/0.1.SETUP-MLIR-OUT-OF-TREE-TEMPLATE.md moved to -> Docs/0.LLVM+MLIR-initial-setup-docs/0.1.SETUP-MLIR-OUT-OF-TREE-TEMPLATE.md 2. README.md updated accordingly
* | 0a0b2a5 Merge branch 'ch-0-1-prep-mlir-template' with 'master'
|\| 
| * 8262069 ch-0-1-prep-mlir-template branch init commit
|/  
*   f5f24d4 Merge branch 'ch-0-0-build-llvm' with 'master'
|\  
| * 2d60e35 (ch-0-0-build-llvm) ch-0-0-build-llvm branch init commit
|/  
* fc52f3e The very first init commit
```

- **With `git-graph` plugin from `VS Code`**

press `ctrl+shift+p` to open the VScode command input box.

Then search/type the command `Git Graph: View Git Graph (git log)`

### 2.1. How to `--delete` branch

- Assume, you want to `--delete` the `ch-0-2-prep-toy-scaffold` branch.
- And also assume, you found the immediate parent branch, which is `master`
- Now first switch to `master` by `git checkout master`
- Now the command to delete is

```sh
git branch --delete ch-0-2-prep-toy-scaffold
#Returns
Deleted branch ch-0-2-prep-toy-scaffold (was 0a0b2a5)
```



## 3. How to use `git cherry-pick <commit-hash>` for updating a small portion of code into another branch.


### Example problem

Let me give you an example case.

In `develop` branch: You have 2 files: `main.cpp` and `header.h`

In `feature` branch: You have 3 files: `main.cpp`, `header.h`, `test.cpp`

Now, you have a **very minor change/patch inserted in some part of the code** of `main.cpp` of `feature` branch.

Now the problem is, **If you want to merge the `feature` with `develop`; `test.cpp` will also be merged to `develop`. And YOU DON'T WANT THAT. You just want to `merge` the little change in `main.cpp` from `feature` branch, to `develop` branch; without affecting `header.h` or `test.cpp` in `develop` branch.**


### Solution:

**3.1. Case-1: For simple scenario**

- checkout `feature` branch: `git checkout feature`

- Insert your patch to `main.cpp`

- Make a commit with only that change. If you have some changes in `header.h`, DONOT stage + commit. Keep it untracked. This commit will only carry the changes/patch in `main.cpp`. Commit message could be something like `minor change in main.cpp for 'feature' branch which would be merged to 'develop' as cherry-pick patch.`

- Now collect the commit hash with `git log` command. Assume the hash is `a3s5da35`.

- Now `checkout` to `develop` branch. `git checkout develop`

- run cherry-pick with the hash you've just collected: `git cherry-pick a3s5da35`. It will give you following message

```sh
Auto-merging main.cpp
[master f722a77] minor change in main.cpp for 'feature' branch which would be merged to 'develop' as cherry-pick patch.
 Date: Fri Dec 29 19:53:07 2023 +0100
 1 file changed, 8 insertions(+), 1 deletion(-)
```

- YOU ARE DONE! 😁


**3.2. Case-2: If you want to `commit` the changes done in `test.cpp` which is not in `develop` branch**

- Then it will become a `merge` commit, and ask you, if you want to resolve `merge conflict` (i.e. will create `test.cpp` in `develop` branch with all the contents). So becareful with this case.


**3.2. Case-2: If the changes/patches in `main.cpp` are already present in `develop` `main.cpp`**

You will see

```sh
Auto-merging main.cpp
On branch develop
You are currently cherry-picking commit a3s5da35.
  (all conflicts fixed: run "git cherry-pick --continue")
  (use "git cherry-pick --skip" to skip this patch)
  (use "git cherry-pick --abort" to cancel the cherry-pick operation)

nothing to commit, working tree clean
The previous cherry-pick is now empty, possibly due to conflict resolution.
If you wish to commit it anyway, use:

    git commit --allow-empty

Otherwise, please use 'git cherry-pick --skip'
```

What I do, allow empty commit, i.e. `git commit --allow-empty`



## 4. How to remove a file using `git` so that it is removed from the earlier commits too

### 4.1. Let's see, what happens if we remove a file using `git rm`

**Let's assume you have a file named `standalone-opt` with size of 160MB. `Github` will not let you commit a file size more than `100MB`. Now you want to remove it with `git rm`. But this command will delete the file in the sense that it will not be tracked by git anymore, but that does not remove the old commit objects corresponding to that file, and so you will still be stuck with pushing the earlier commits which correspond to that `160MB` of `standalone-opt`.**


### 4.2. Solution

Read this stackoverflow thread [git rm - fatal: pathspec did not match any files](https://stackoverflow.com/a/25458504)/
