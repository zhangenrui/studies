#!/usr/bin/env bash

repo_path=$(pwd)
algo="algorithm"
src="src"
src_path="$repo_path/$src"
scripts_path="$repo_path/scripts"

#echo "$src_path"
#echo "$algo_path"
#echo "$scripts_path"

#cd "$repo" || exit

# 执行文档测试
printf "=== Start Doc Test ===\n"
out=$(python "$scripts_path/doctest_modules_recur.py")
if [[ $out != 0 ]]; then
  echo "Not all file pass doctest($src_path)"
  exit
else
  echo "$out"
  echo "All file passed doctest."
fi
echo

# 生成 README.md
printf "=== Start generating README.md ===\n"
out=$(python "$scripts_path/generate_readme_files.py")
echo "$out"
if [[ $out != 0 ]]; then
  git commit -m "$(git log -1 --pretty=%B | cat)"  # 使用上一次的提交信息
  # git commit -m "$(git log -"$(git rev-list origin/master..master --count)" --pretty=%B | cat)"  # 用最近N次的提交信息作为自动提交的信息
fi
echo

# 主仓库
printf "=== Start Push Main Repo ===\n"
git push
echo

# 子仓库（单目录） algorithm
#name="algorithm"
#echo "=== Start Push $name ==="
#git subtree split --prefix=$algo --branch $name --rejoin
#git subtree push --prefix=$algo $name master --squash
#echo

# 子仓库（多目录） pytorch_trainer
#name="pytorch_trainer"
#echo "=== Start Push $name ==="
#sh "$scripts_path/create_sub_repo.sh" $name
#echo


# 统计 commit 次数
#num_commits=$(git log | grep '^commit ' | awk '{print $1}' | uniq -c | awk '{print $1}')
#split_feq=20  # 每提交 20 次再 split 一次
#split_flag=$((num_commits % split_feq))


#prefix="code"
#name="my_lab"
#echo "=== Start Push $name ==="
#git subtree split --prefix=$prefix --branch $name --rejoin
#git subtree push --prefix=$prefix $name master --squash
#echo
#
#prefix="code/my"
#name="my"
#echo "=== Start Push $name ==="
#git subtree split --prefix=$prefix --branch $name --rejoin
#git subtree push --prefix=$prefix $name master --squash
#echo


#====================== history

# git subtree add --prefix=code/keras_demo/keras_model/bert_by_keras bert_by_keras master --squash
# git subtree add --prefix=code/keras_demo keras_demo master --squash

# 使用 submodule 代替 subtree
# git subtree push --prefix=code/keras_demo/keras_model/bert_by_keras bert_by_keras master
# git subtree push --prefix=code/keras_demo keras_demo master

# 获取仓库父目录
#pwd=$(pwd)

# 先更新子仓库
#printf "=== First: Update submodule ===\n"

# 1.
#sub_repo="bert_by_keras"
#echo "____ Start update $sub_repo"
#cd "$pwd/code/my_models/$sub_repo" || exit
#ret=$(git pull origin master)
#if [[ $ret =~ "Already up to date" ]]; then
#  echo "$sub_repo is already up to date."
#else
#  cd "$pwd" || exit
#  git add "$pwd/code/my_models/$sub_repo"
#  git commit -m "U $sub_repo"
#fi

# 更新父仓库
#cd "$pwd" || exit
#printf "\n=== Final: Push father repository ===\n"
#git push
