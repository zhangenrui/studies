
studies=$(pwd)
repo_name=$1
repo="$HOME/tmp/$repo_name"

rm -rf "$repo"
mkdir -p "$repo/src"

cd "$repo" || exit
git init

# .gitignore
cp "$studies/.gitignore" "$repo/.gitignore"
cp -R "$studies/src/$repo_name" "$repo/src/$repo_name"
cp -R "$studies/src/huaytools" "$repo/src/huaytools"  # 工具库
cp -R "$studies/examples/$repo_name" "$repo/examples"  # 示例
cp "$studies/src/$repo_name/README.md" "$repo/README.md"  # README.md

git add .
git commit -q -m 'Init'
git remote add origin "https://github.com/imhuay/$repo_name.git"
git push --force --set-upstream origin master

#rm -rf "$repo"
