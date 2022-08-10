    # $1: $(target_path) 
    # $2: $(TARGET_BRANCH) - $(System.PullRequest.TargetBranch)
    # $3: $(VAL_PATH) 

    supported_extensions=(py, sh, yaml)
                    
    set -xe

    git --no-pager diff --name-only $(git show-ref -s remotes/origin/$2) ./$1 > $3/copyright/diff.log

    files=$(cat $3/copyright/diff.log | awk '!a[$0]++')

    for file in ${files}

    do
        if [[ "${supported_extensions[@]}" =~ "${file##*.}" ]]; then
            echo "Checking license in 1 ${file}"
            if [ $(grep -E -c "Copyright \\(c\\) ([0-9]{4})(-[0-9]{4})? Intel Corporation" ${file}) = 0 ]; then
                echo ${file} >>  $3/copyright/copyright_issue_summary.log
                cat $3/copyright/copyright_issue_summary.log
            fi
        else
            echo "Skipping ${file}"
        fi
    done


    ls $3/copyright/copyright_issue_summary.log
    exit_code=$?
    if [ ${exit_code} -e 0 ] ; then
        echo "------------------Check <copyright_issue_summary.log> for wrong file list !!!!!!!!!!!!!!!!!!!!!!!"; exit 1
    fi
    exit 0