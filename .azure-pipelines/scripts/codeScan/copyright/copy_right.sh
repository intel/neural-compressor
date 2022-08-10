    # $1: $(target_path) 
    # $2: $(TARGET_BRANCH) - $(System.PullRequest.TargetBranch)
    # $3: $(VAL_PATH)
    
    pip install -r requirements.txt


    supported_extensions=(py, sh, yaml)
                    
    set -xe
    git config --global --add safe.directory /neural_compressor
    git --no-pager diff --name-only --no-index $(git show-ref -s remotes/origin$2) /neural_compressor/neural_compressor > /diff.log
    # git --no-pager diff --name-only $(git show-ref -s remotes/origin$2) ./$1 > $3/copyright/diff.log

    files=$(cat /diff.log | awk '!a[$0]++')
    # files=$(cat $3/copyright/diff.log | awk '!a[$0]++')

    for file in ${files}

    do
        if [[ "${supported_extensions[@]}" =~ "${file##*.}" ]]; then
            echo "Checking license in 1 ${file}"
            if [ $(grep -E -c "Copyright \\(c\\) ([0-9]{4})(-[0-9]{4})? Intel Corporation" ${file}) = 0 ]; then
                echo ${file} >>  /copyright_issue_summary.log
                cat /copyright_issue_summary.log
            fi
        else
            echo "Skipping ${file}"
        fi
    done


    ls /copyright_issue_summary.log
    exit_code=$?
    if [ ${exit_code} -e 0 ] ; then
        echo "------------------Check <copyright_issue_summary.log> for wrong file list !!!!!!!!!!!!!!!!!!!!!!!"; exit 1
    fi
    exit 0