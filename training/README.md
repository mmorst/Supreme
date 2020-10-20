## Git/Github

### download data on your computer
In the command line (powershell on Windows, terminal on Mac), go to the folder where you want to create your repository and type:
`git clone https://github.com/mmorst/Supreme.git`

### push data to the server
`git add .`

`git commit -m "description of the commit"`

Now you want to update you local version and merge with the latest:

`git fetch origin`

`git merge origin/master`


If ther merge is successful, then you can do:

`git push origin`

## Coding tips

When tinkering with jupyter notebooks, please make a copy - there are often merge problems when we edit those.