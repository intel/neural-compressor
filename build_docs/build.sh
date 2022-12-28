CHECKOUT_GH_PAGES=1
PUSH_GH_PAGES=1
WORK_DIR=../build_tmp
rm -rf ${WORK_DIR}
mkdir -p ${WORK_DIR}
cp -rf ./* ${WORK_DIR}

if [ ! -d env_sphinx ]; then
    bash pip_set_env.sh
fi
source env_sphinx/bin/activate

cd ${WORK_DIR}
cp -rf ../docs/ ./source

cp -f "../README.md" "./source/docs/source/Welcome.md"
cp -f "../SECURITY.md" "./source/docs/source/SECURITY.md"
cp ../neural_coder/extensions/screenshots/* ./source/docs/source/imgs

sed -i 's/.\/neural_coder\/extensions\/screenshots/imgs/g' ./source/docs/source/Welcome.md

sed -i 's/.\/docs\/source\/_static/./g' ./source/docs/source/Welcome.md

sed -i 's/.md/.html/g; s/.\/docs\/source\//.\//g' ./source/docs/source/Welcome.md


make clean
make html

if [[ $? -eq 0 ]]; then
  echo "Sphinx build online documents successfully!"
else
  echo "Sphinx build online documents fault!"
  exit 1
fi

if [[ ${CHECKOUT_GH_PAGES} -eq 1 ]]; then
  git checkout -b gh-pages
  git branch --set-upstream-to=origin/gh-pages gh-pages
  git pull
  git fetch origin
  git reset --hard origin/gh-pages
else
  echo "skip pull gh-pages"
fi


VERSION=`cat source/version.txt`
DST_FOLDER=../${VERSION}
LATEST_FOLDER=../latest
SRC_FOLDER=build/html

rm -rf ${DST_FOLDER}/*
mkdir -p ${DST_FOLDER}
cp -r ${SRC_FOLDER}/* ${DST_FOLDER}
python update_html.py ${DST_FOLDER} ${VERSION}
cp -r ./source/docs/source/imgs ${DST_FOLDER}/docs/source


rm -rf ${LATEST_FOLDER}/*
mkdir -p ${LATEST_FOLDER}
cp -r ${SRC_FOLDER}/* ${LATEST_FOLDER}
python update_html.py ${LATEST_FOLDER} ${VERSION}
cp -r ./source/docs/source/imgs ${LATEST_FOLDER}/docs/source

echo "Create document is done"

if [[ ${PUSH_GH_PAGES} -eq 1 ]]; then
  git add ${LATEST_FOLDER} ${DST_FOLDER} ../versions.html
  git commit -m "update for ${VERSION}"
  git push origin gh-pages
else
  echo "Skip push"	
fi

if [[ $? -eq 0 ]]; then
  echo "push online documents successfully!"
else
  echo "push build online documents fault!"
  exit 1
fi

exit 0
