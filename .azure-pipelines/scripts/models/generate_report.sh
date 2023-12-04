#!/bin/bash

# WORKSPACE=.
# summaryLog=summary.log
# summaryLogLast=summary.log
# tuneLog=tuning_info.log
# tuneLogLast=tuning_info.log
# overview_log=summary_overview.log
# coverage_summary=coverage_summary.log
# nc_code_lines_summary=nc_code_lines_summary.csv
# engine_code_lines_summary=engine_code_lines_summary.csv

#lines_coverage_threshold=80
#branches_coverage_threshold=75
#
#pass_status="<td style=\"background-color:#90EE90\">Pass</td>"
#fail_status="<td style=\"background-color:#FFD2D2\">Fail</td>"
#verify_status="<td style=\"background-color:#f2ea0a\">Verify</td>"


# shellcheck disable=SC2120

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    -w|--WORKSPACE)
        WORKSPACE=${2}
        shift 2
        ;;
    --script_path)
        script_path=${2}
        shift 2
        ;;
    --output_dir)
        output_dir=${2}
        shift 2
        ;;
    --last_logt_dir)
        last_logt_dir=${2}
        shift 2
        ;;
    *)
        shift
        ;;
  esac
done

echo "workspace: ${WORKSPACE}"
echo "script_path: ${script_path}"

summaryLog="${WORKSPACE}/summary.log"
tuneLog="${WORKSPACE}/tuning_info.log"
echo "summaryLog: ${summaryLog}"
echo "tuneLog: ${tuneLog}"

echo "last_logt_dir: ${last_logt_dir}"
summaryLogLast="${last_logt_dir}/summary.log"
tuneLogLast="${last_logt_dir}/tuning_info.log"
echo "summaryLogLast: ${summaryLogLast}"
echo "tuneLogLast: ${tuneLogLast}"
ghprbPullId=${SYSTEM_PULLREQUEST_PULLREQUESTNUMBER}
MR_source_branch=${SYSTEM_PULLREQUEST_SOURCEBRANCH}
MR_source_repo=${SYSTEM_PULLREQUEST_SOURCEREPOSITORYURI}
MR_target_branch=${SYSTEM_PULLREQUEST_TARGETBRANCH}
repo_url=${BUILD_REPOSITORY_URI}
source_commit_id=${BUILD_SOURCEVERSION}
build_id=${BUILD_BUILDID}
echo "MR_source_branch: ${MR_source_branch}"
echo "MR_source_repo: ${MR_source_repo}"
echo "MR_target_branch: ${MR_target_branch}"
echo "repo_url: ${repo_url}"
echo "commit_id: ${source_commit_id}"
echo "ghprbPullId: ${ghprbPullId}"
echo "build_id: ${build_id}"


function main {
    generate_html_head
    generate_html_body
    generate_results
    generate_html_footer
}

function generate_inference {
#     echo "Generating inference"
    awk -v framework="${framework}" -v fw_version="${fw_version}" -v model="${model}" -v os="${os}" -v platform=${platform} -F ';' '
        BEGINE {
            fp32_perf_bs = "nan";
            fp32_perf_value = "nan";
            fp32_perf_url = "nan";
            fp32_acc_bs = "nan";
            fp32_acc_value = "nan";
            fp32_acc_url = "nan";

            int8_perf_bs = "nan";
            int8_perf_value = "nan";
            int8_perf_url = "nan";
            int8_acc_bs = "nan";
            int8_acc_value = "nan";
            int8_acc_url = "nan";
        }{
            if($1 == os && $2 == platform && $3 == framework && $4 == fw_version && $6 == model) {
                // FP32
                if($5 == "FP32") {
                    // Performance
                    if($8 == "Performance") {
                        fp32_perf_bs = $9;
                        fp32_perf_value = $10;
                        fp32_perf_url = $11;
                    }
                    // Accuracy
                    if($8 == "Accuracy") {
                        fp32_acc_bs = $9;
                        fp32_acc_value = $10;
                        fp32_acc_url = $11;
                    }
                }

                // INT8
                if($5 == "INT8") {
                    // Performance
                    if($8 == "Performance") {
                        int8_perf_bs = $9;
                        int8_perf_value = $10;
                        int8_perf_url = $11;
                    }
                    // Accuracy
                    if($8 == "Accuracy") {
                        int8_acc_bs = $9;
                        int8_acc_value = $10;
                        int8_acc_url = $11;
                    }
                }
            }
        }END {
            printf("%s;%s;%s;%s;", int8_perf_bs,int8_perf_value,int8_acc_bs,int8_acc_value);
            printf("%s;%s;%s;%s;", fp32_perf_bs,fp32_perf_value,fp32_acc_bs,fp32_acc_value);
            printf("%s;%s;%s;%s;", int8_perf_url,int8_acc_url,fp32_perf_url,fp32_acc_url);
        }
    ' "$1"
}

function generate_html_core {
    echo "--- current values ---"
    echo ${current_values}
    echo "--- last values ---"
    echo ${last_values}
    tuning_strategy=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLog} |awk -F';' '{print $6}')
    tuning_time=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLog} |awk -F';' '{print $7}')
    tuning_count=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLog} |awk -F';' '{print $8}')
    tuning_log=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLog} |awk -F';' '{print $9}')
    echo "<tr><td rowspan=3>${platform}</td><td rowspan=3>${os}</td><td rowspan=3>${framework}</td><td rowspan=3>${fw_version}</td><td rowspan=3>${model}</td><td>New</td><td><a href=${tuning_log}>${tuning_strategy}</a></td>" >> ${output_dir}/report.html
    echo "<td><a href=${tuning_log}>${tuning_time}</a></td><td><a href=${tuning_log}>${tuning_count}</a></td>" >> ${output_dir}/report.html

    tuning_strategy=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLogLast} |awk -F';' '{print $6}')
    tuning_time=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLogLast} |awk -F';' '{print $7}')
    tuning_count=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLogLast} |awk -F';' '{print $8}')
    tuning_log=$(grep "^${os};${platform};${framework};${fw_version};${model};" ${tuneLogLast} |awk -F';' '{print $9}')

    echo |awk -F ';' -v current_values="${current_values}" -v last_values="${last_values}" \
              -v tuning_strategy="${tuning_strategy}" -v tuning_time="${tuning_time}" \
              -v tuning_count="${tuning_count}" -v tuning_log="${tuning_log}" -F ';' '

        function abs(x) { return x < 0 ? -x : x }

        function show_new_last(batch, link, value, metric) {
            if(value ~/[1-9]/) {
                if (metric == "perf" || metric == "ratio") {
                    printf("<td>%s</td> <td><a href=%s>%.2f</a></td>\n",batch,link,value);
                } else {
                    printf("<td>%s</td> <td><a href=%s>%.2f%</a></td>\n",batch,link,value*100);
                }
            } else {
                if(link == "" || value == "N/A" || value == "unknown") {
                    printf("<td></td> <td></td>\n");
                } else {
                    printf("<td>%s</td> <td><a href=%s>Failure</a></td>\n",batch,link);
                }
            }
        }

        function compare_current(int8_result, fp32_result, metric) {

            if(int8_result ~/[1-9]/ && fp32_result ~/[1-9]/) {
                if(metric == "acc") {
                    target = (int8_result - fp32_result) / fp32_result;
                    if(target >= -0.01) {
                        printf("<td rowspan=3 style=\"background-color:#90EE90\">%.2f %</td>", target*100);
                    }else if(target < -0.05) {
                        printf("<td rowspan=3 style=\"background-color:#FFD2D2\">%.2f %</td>", target*100);
                        job_status = "fail"
                    }else{
                        printf("<td rowspan=3>%.2f %</td>", target*100);
                    }
                }else if(metric == "perf") {
                    target = int8_result / fp32_result;
                    if(target >= 1.5) {
                        printf("<td style=\"background-color:#90EE90\">%.2f</td>", target);
                    }else if(target < 1) {
                        printf("<td style=\"background-color:#FFD2D2\">%.2f</td>", target);
                        perf_status = "fail"
                    }else{
                        printf("<td>%.2f</td>", target);
                    }
                }
                else {
                    target = int8_result / fp32_result;
                    if(target >= 2) {
                        printf("<td rowspan=3 style=\"background-color:#90EE90\">%.2f</td>", target);
                    }else if(target < 1) {
                        printf("<td rowspan=3 style=\"background-color:#FFD2D2\">%.2f</td>", target);
                        job_status = "fail"
                    }else{
                        printf("<td rowspan=3>%.2f</td>", target);
                    }
                }
            }else {
                printf("<td rowspan=3></td>");
            }
        }

        function compare_result(new_result, previous_result, metric) {

            if (new_result ~/[1-9]/ && previous_result ~/[1-9]/) {
                if(metric == "acc") {
                    target = new_result - previous_result;
                    if(target > -0.00001 && target < 0.00001) {
                        status_png = "background-color:#90EE90";
                    } else {
                        status_png = "background-color:#FFD2D2";
                        job_status = "fail"
                    }
                    printf("<td style=\"%s\" colspan=2>%.2f %</td>", status_png, target*100);
                } else {
                    target = new_result / previous_result;
                    if(target <= 1.084 && target >= 0.915) {
                        status_png = "background-color:#90EE90";
                    } else {
                        status_png = "background-color:#FFD2D2";
                        perf_status = "fail"
                    }
                    printf("<td style=\"%s\" colspan=2>%.2f</td>", status_png, target);
                }
            } else {
              if((new_result == nan && previous_result == nan) || new_result == "unknown"){
                    printf("<td class=\"col-cell col-cell3\" colspan=2></td>");
              } else{
                  if(new_result == nan) {
                        job_status = "fail"
                        status_png = "background-color:#FFD2D2";
                        printf("<td style=\"%s\" colspan=2></td>", status_png);
                  } else{
                        printf("<td class=\"col-cell col-cell3\" colspan=2></td>");
                  }
              }
            }
        }

        function compare_ratio(int8_perf_value, fp32_perf_value, last_int8_perf_value, last_fp32_perf_value) {
            if (int8_perf_value ~/[1-9]/ && fp32_perf_value ~/[1-9]/ && last_int8_perf_value ~/[1-9]/ && last_fp32_perf_value ~/[1-9]/) {
                new_result = int8_perf_value / fp32_perf_value
                previous_result = last_int8_perf_value / last_fp32_perf_value
                target = new_result / previous_result;
                if (target <= 1.084 && target >= 0.915) {
                    status_png = "background-color:#90EE90";
                } else {
                    status_png = "background-color:#FFD2D2";
                    ratio_status = "fail"
                }
                printf("<td style=\"%s\">%.2f</td>", status_png, target);
            } else {
                if (new_result == nan && previous_result == nan) {
                    printf("<td class=\"col-cell col-cell3\"></td>");
                } else {
                    if (new_result == nan) {
                        ratio_status = "fail"
                        status_png = "background-color:#FFD2D2";
                        printf("<td style=\"%s\"></td>", status_png);
                    } else {
                        printf("<td class=\"col-cell col-cell3\"></td>");
                    }
                }
            }
        }

        BEGIN {
            job_status = "pass"
            perf_status = "pass"
            ratio_status = "pass"
            // issue list
            jira_mobilenet = "https://jira01.devtools.intel.com/browse/PADDLEQ-384";
            jira_resnext = "https://jira01.devtools.intel.com/browse/PADDLEQ-387";
            jira_ssdmobilenet = "https://jira01.devtools.intel.com/browse/PADDLEQ-406";
        }{
            // Current values
            split(current_values,current_value,";");

            // Current

            // INT8 Performance results
            int8_perf_batch=current_value[1]
            int8_perf_value=current_value[2]
            int8_perf_url=current_value[9]
            show_new_last(int8_perf_batch, int8_perf_url, int8_perf_value, "perf");

            // INT8 Accuracy results
            int8_acc_batch=current_value[3]
            int8_acc_value=current_value[4]
            int8_acc_url=current_value[10]
            show_new_last(int8_acc_batch, int8_acc_url, int8_acc_value, "acc");

             // FP32 Performance results
            fp32_perf_batch=current_value[5]
            fp32_perf_value=current_value[6]
            fp32_perf_url=current_value[11]
            show_new_last(fp32_perf_batch, fp32_perf_url, fp32_perf_value, "perf");

            // FP32 Accuracy results
            fp32_acc_batch=current_value[7]
            fp32_acc_value=current_value[8]
            fp32_acc_url=current_value[12]
            show_new_last(fp32_acc_batch, fp32_acc_url, fp32_acc_value, "acc");

            // Compare Current

            compare_current(int8_perf_value, fp32_perf_value, "perf");
            compare_current(int8_acc_value, fp32_acc_value, "acc");

            // Last values
            split(last_values,last_value,";");

            // Last
            printf("</tr>\n<tr><td>Last</td><td><a href=%4$s>%1$s</a></td><td><a href=%4$s>%2$s</a></td><td><a href=%4$s>%3$s</a></td>", tuning_strategy, tuning_time, tuning_count, tuning_log);

             // Show last INT8 Performance results
            last_int8_perf_batch=last_value[1]
            last_int8_perf_value=last_value[2]
            last_int8_perf_url=last_value[9]
            show_new_last(last_int8_perf_batch, last_int8_perf_url, last_int8_perf_value, "perf");

            // Show last INT8 Accuracy results
            last_int8_acc_batch=last_value[3]
            last_int8_acc_value=last_value[4]
            last_int8_acc_url=last_value[10]
            show_new_last(last_int8_acc_batch, last_int8_acc_url, last_int8_acc_value, "acc");

            // Show last FP32 Performance results
            last_fp32_perf_batch=last_value[5]
            last_fp32_perf_value=last_value[6]
            last_fp32_perf_url=last_value[11]
            show_new_last(last_fp32_perf_batch, last_fp32_perf_url, last_fp32_perf_value, "perf");

            // Show last FP32 Accuracy results
            last_fp32_acc_batch=last_value[7]
            last_fp32_acc_value=last_value[8]
            last_fp32_acc_url=last_value[12]
            show_new_last(last_fp32_acc_batch, last_fp32_acc_url, last_fp32_acc_value, "acc");

            compare_current(last_int8_perf_value, last_fp32_perf_value, "perf");

            printf("</tr>")

            // current vs last
            printf("</tr>\n<tr><td>New/Last</td><td colspan=3 class=\"col-cell3\"></td>");

            // Compare INT8 Performance results
            compare_result(int8_perf_value, last_int8_perf_value,"perf");

            // Compare INT8 Accuracy results
            compare_result(int8_acc_value, last_int8_acc_value, "acc");

            // Compare FP32 Performance results
            compare_result(fp32_perf_value, last_fp32_perf_value, "perf");

            // Compare FP32 Accuracy results
            compare_result(fp32_acc_value, last_fp32_acc_value, "acc");

            // Compare INT8 FP32 Performance ratio
            compare_ratio(int8_perf_value, fp32_perf_value, last_int8_perf_value, last_fp32_perf_value);

            printf("</tr>\n");

            status = (perf_status == "fail" && ratio_status == "fail") ? "fail" : "pass"
            status = (job_status == "fail") ? "fail" : status

        } END{
            printf("\n%s", status);
        }
    ' >> ${output_dir}/report.html
    job_state=$(tail -1 ${WORKSPACE}/report.html)
    sed -i '$s/.*//' ${WORKSPACE}/report.html

    if [ ${job_state} == 'fail' ]; then
      echo "====== perf_reg ======"
      echo "##vso[task.setvariable variable=is_perf_reg]true"
    fi
}

function generate_results {
    echo "Generating tuning results"
    oses=$(sed '1d' ${summaryLog} |cut -d';' -f1 | awk '!a[$0]++')
    echo ${oses}

    for os in ${oses[@]}
    do
        platforms=$(sed '1d' ${summaryLog} |grep "^${os}" |cut -d';' -f2 | awk '!a[$0]++')
        echo ${platforms}
        for platform in ${platforms[@]}
        do
            frameworks=$(sed '1d' ${summaryLog} |grep "^${os};${platform}" |cut -d';' -f3 | awk '!a[$0]++')
            echo ${frameworks}
            for framework in ${frameworks[@]}
            do
                fw_versions=$(sed '1d' ${summaryLog} |grep "^${os};${platform};${framework}" |cut -d';' -f4 | awk '!a[$0]++')
                echo ${fw_versions}
                for fw_version in ${fw_versions[@]}
                do
                    models=$(sed '1d' ${summaryLog} |grep "^${os};${platform};${framework};${fw_version}" |cut -d';' -f6 | awk '!a[$0]++')
                    echo ${models}
                    for model in ${models[@]}
                    do
                        echo "--- processing model ---"
                        echo ${model}
                        current_values=$(generate_inference ${summaryLog})
                        echo "| current value |"
                        echo ${current_values}
                        last_values=$(generate_inference ${summaryLogLast})
                        echo "| last value |"
                        echo ${last_values}

                        generate_html_core ${current_values} ${last_values}
                    done
                done
            done
        done
    done
}

function generate_html_body {
MR_TITLE=''
Test_Info_Title=''
Test_Info=''

if [ "${qtools_branch}" == "" ];
then
    commit_id=$(echo ${ghprbActualCommit} |awk '{print substr($1,1,7)}')

    MR_TITLE="[ <a href='${repo_url}/pull/${ghprbPullId}'>PR-${ghprbPullId}</a> ]"
    Test_Info_Title="<th colspan="2">Source Branch</th> <th colspan="4">Target Branch</th> <th colspan="4">Commit</th> "
    Test_Info="<td colspan="2">${MR_source_branch}</td> <td colspan="4"><a href='${repo_url}/tree/${MR_target_branch}'>${MR_target_branch}</a></td> <td colspan="4"><a href='${MR_source_repo}/commit/${source_commit_id}'>${source_commit_id:0:6}</a></td>"
else
    Test_Info_Title="<th colspan="4">Test Branch</th> <th colspan="4">Commit ID</th> "
    Test_Info="<th colspan="4">${qtools_branch}</th> <th colspan="4">${qtools_commit}</th> "
fi

cat >> ${output_dir}/report.html << eof

<body>
    <div id="main">
        <h1 align="center">Neural Compressor Tuning Tests ${MR_TITLE}
            [ <a
                href="https://dev.azure.com/lpot-inc/neural-compressor/_build/results?buildId=${build_id}">Job-${build_id}</a>
            ]</h1>
        <h1 align="center">Test Status: ${Jenkins_job_status}</h1>
        <h2>Summary</h2>
        <table class="features-table">
            <tr>
                <th>Repo</th>
                ${Test_Info_Title}
            </tr>
            <tr>
                <td><a href="https://github.com/intel/neural-compressor">neural-compressor</a></td>
                ${Test_Info}
            </tr>
        </table>
eof


echo "Generating benchmarks table"
cat >> ${output_dir}/report.html << eof
        <h2>Benchmark</h2>
          <table class="features-table">
            <tr>
                <th rowspan="2">Platform</th>
                <th rowspan="2">System</th>
                <th rowspan="2">Framework</th>
                <th rowspan="2">Version</th>
                <th rowspan="2">Model</th>
                <th rowspan="2">VS</th>
                <th rowspan="2">Tuning<br>Strategy</th>
                <th rowspan="2">Tuning<br>Time(s)</th>
                <th rowspan="2">Tuning<br>Count</th>
                      <th colspan="4">INT8</th>
                      <th colspan="4">FP32</th>
                      <th colspan="2" class="col-cell col-cell1 col-cellh">Ratio</th>
                </tr>
                <tr>

                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th class="col-cell col-cell1">Throughput<br><font size="2px">INT8/FP32</font></th>
                <th class="col-cell col-cell1">Accuracy<br><font size="2px">(INT8-FP32)/FP32</font></th>
                </tr>
eof
}

function generate_html_footer {

    cat >> ${output_dir}/report.html << eof
            <tr>
                <td colspan="22"><font color="#d6776f">Note: </font>All data tested on Azure Cloud.</td>
                <td colspan="3" class="col-cell col-cell1 col-cellf"></td>
            </tr>
        </table>
    </div>
</body>
</html>
eof
}

function generate_html_head {

cat > ${output_dir}/report.html << eof

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html lang="en">
<head>
    <meta http-equiv="content-type" content="text/html; charset=ISO-8859-1">
    <title>Daily Tests - TensorFlow - Jenkins</title>
    <style type="text/css">
        body
        {
            margin: 0;
            padding: 0;
            background: white no-repeat left top;
        }
        #main
        {
            // width: 100%;
            margin: 20px auto 10px auto;
            background: white;
            -moz-border-radius: 8px;
            -webkit-border-radius: 8px;
            padding: 0 30px 30px 30px;
            border: 1px solid #adaa9f;
            -moz-box-shadow: 0 2px 2px #9c9c9c;
            -webkit-box-shadow: 0 2px 2px #9c9c9c;
        }
        .features-table
        {
            width: 100%;
            margin: 0 auto;
            border-collapse: separate;
            border-spacing: 0;
            text-shadow: 0 1px 0 #fff;
            color: #2a2a2a;
            background: #fafafa;
            background-image: -moz-linear-gradient(top, #fff, #eaeaea, #fff); /* Firefox 3.6 */
            background-image: -webkit-gradient(linear,center bottom,center top,from(#fff),color-stop(0.5, #eaeaea),to(#fff));
            font-family: Verdana,Arial,Helvetica
        }
        .features-table th,td
        {
            text-align: center;
            height: 25px;
            line-height: 25px;
            padding: 0 8px;
            border: 1px solid #cdcdcd;
            box-shadow: 0 1px 0 white;
            -moz-box-shadow: 0 1px 0 white;
            -webkit-box-shadow: 0 1px 0 white;
            white-space: nowrap;
        }
        .no-border th
        {
            box-shadow: none;
            -moz-box-shadow: none;
            -webkit-box-shadow: none;
        }
        .col-cell
        {
            text-align: center;
            width: 150px;
            font: normal 1em Verdana, Arial, Helvetica;
        }
        .col-cell3
        {
            background: #efefef;
            background: rgba(144,144,144,0.15);
        }
        .col-cell1, .col-cell2
        {
            background: #B0C4DE;
            background: rgba(176,196,222,0.3);
        }
        .col-cellh
        {
            font: bold 1.3em 'trebuchet MS', 'Lucida Sans', Arial;
            -moz-border-radius-topright: 10px;
            -moz-border-radius-topleft: 10px;
            border-top-right-radius: 10px;
            border-top-left-radius: 10px;
            border-top: 1px solid #eaeaea !important;
        }
        .col-cellf
        {
            font: bold 1.4em Georgia;
            -moz-border-radius-bottomright: 10px;
            -moz-border-radius-bottomleft: 10px;
            border-bottom-right-radius: 10px;
            border-bottom-left-radius: 10px;
            border-bottom: 1px solid #dadada !important;
        }
    </style>
</head>

eof

}

main
