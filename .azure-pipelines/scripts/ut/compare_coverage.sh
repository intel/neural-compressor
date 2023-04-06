output_file=$1
coverage_pr_log=$2
coverage_base_log=$3
coverage_status=$4
coverage_PR_lines_rate=$5
coverage_base_lines_rate=$6
coverage_PR_branches_rate=$7
coverage_base_branches_rate=$8
module_name="neural_compressor"
[[ ! -f $coverage_pr_log ]] && exit 1
[[ ! -f $coverage_base_log ]] && exit 1
file_name="./coverage_compare"
sed -i "s|\/usr.*${module_name}\/||g" $coverage_pr_log
sed -i "s|\/usr.*${module_name}\/||g" $coverage_base_log
diff $coverage_pr_log $coverage_base_log > diff_file
[[ $? == 0 ]] && exit 0
grep -Po "[<,>,\d].*" diff_file | awk '{print $1 "\t" $2 "\t" $3 "\t"  $4 "\t"  $5 "\t" $6 "\t" $7}' | sed "/Name/d" | sed "/TOTAL/d" |sed "/---/d" > $file_name
[[ ! -s $file_name ]] && exit 0
[[ -f $output_file ]] && rm -f $output_file
touch $output_file

function generate_html_head {

cat > ${output_file} << eof

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UT coverage</title>
    <style type="text/css">
        body {
            margin: 0;
            padding: 0;
            background: white no-repeat left top;
        }

        #main {
            margin: 20px auto 10px auto;
            background: white;
            border-radius: 8px;
            -moz-border-radius: 8px;
            -webkit-border-radius: 8px;
            padding: 0 30px 30px 30px;
            border: 1px solid #adaa9f;
            box-shadow: 0 2px 2px #9c9c9c;
            -moz-box-shadow: 0 2px 2px #9c9c9c;
            -webkit-box-shadow: 0 2px 2px #9c9c9c;
        }

        .features-table {
            width: 100%;
            margin: 0 auto;
            border-collapse: separate;
            border-spacing: 0;
            text-shadow: 0 1px 0 #fff;
            color: #2a2a2a;
            background: #fafafa;
            background-image: -moz-linear-gradient(top, #fff, #eaeaea, #fff);
            /* Firefox 3.6 */
            background-image: -webkit-gradient(linear, center bottom, center top, from(#fff), color-stop(0.5, #eaeaea), to(#fff));
            font-family: Verdana, Arial, Helvetica
        }

        .features-table th,
        td {
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
    </style>
</head>

eof
}

function main {
    generate_html_head
    # generate table head
    Lines_cover_decrease=$(echo "$coverage_PR_lines_rate - $coverage_base_lines_rate" | bc -l)
    Branches_cover_decrease=$(echo "$coverage_PR_branches_rate - $coverage_base_branches_rate" | bc -l)

    if (( $(echo "$Lines_cover_decrease < 0" | bc -l) )); then
      lines_coverage_color="#FFD2D2"
    else
      lines_coverage_color="#90EE90"
    fi

    if (( $(echo "$Branches_cover_decrease < 0" | bc -l) )); then
      branches_coverage_color="#FFD2D2"
    else
      branches_coverage_color="#90EE90"
    fi

    echo """
    <body>
        <div id="main">
        <h1 align="center">Coverage Summary : ${coverage_status}</h1>
        <table class=\"features-table\" style=\"width: 60%;margin-left:auto;margin-right:auto;empty-cells: hide\">
            <tr>
                <th></th>
                <th>Base coverage</th>
                <th>PR coverage</th>
                <th>Diff</th>
            </tr>
            <tr>
                <td> Lines </td>
                <td> ${coverage_base_lines_rate}% </td>
                <td> ${coverage_PR_lines_rate}% </td>
                <td style=\"background-color:${lines_coverage_color}\"> ${Lines_cover_decrease}% </td>
            </tr>
            <tr>
                <td> Branches </td>
                <td> ${coverage_base_branches_rate}% </td>
                <td> ${coverage_PR_branches_rate}% </td>
                <td style=\"background-color:${branches_coverage_color}\"> ${Branches_cover_decrease}% </td>
            </tr>
        </table>
        </div>
    """ >> ${output_file}
    if [[ ${coverage_status} = "SUCCESS" ]]; then
      echo """</body></html>""" >> ${output_file}
      echo "coverage PASS, no need to compare difference"
      exit 0
    fi
    echo """
    <div id="main">
        <h2 align="center">Coverage Detail</h2>
        <table class=\"features-table\" style=\"width: 60%;margin-left:auto;margin-right:auto;empty-cells: hide\">
            <tr>
                <th>Commit</th>
                <th>FileName</th>
                <th>Miss</th>
                <th>Branch</th>
                <th>Cover</th>
            </tr>
    """ >> ${output_file}
    # generate compare detail
    cat $file_name | while read line
    do
        if [[ $(echo $line | grep "[0-9]a[0-9]") ]] && [[ $(grep -A 1 "$line" $file_name | grep ">") ]]; then
            diff_lines=$(sed -n "/${line}/,/^[0-9]/p" ${file_name} | grep ">")
            diff_file_name=$(sed -n "/${line}/,/^[0-9]/p" ${file_name} | grep -Po ">.*[a-z,A-Z].*.py" | sed "s|>||g")
            for diff_file in ${diff_file_name}
            do          
              diff_file=$(echo "${diff_file}" | sed 's/[ \t]*//g')
              file=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $1}')
              miss=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $3}')
              cover=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $6}')
              branch=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $4}')
              echo """
              <tr><td>PR | BASE</td><td style=\"text-align:left\">${file}</td>
                  <td style=\"text-align:left\">NA | ${miss}</td>
                  <td style=\"text-align:left\">NA | ${branch}</td>
                  <td style=\"text-align:left\">NA | ${cover}</td>
              </tr>""" >> ${output_file}
            done
        elif [[ $(echo $line | grep "[0-9]c[0-9]") ]] && [[ $(cat $file_name | grep -A 1 "$line" | grep "<") ]]; then
            diff_lines=$(sed -n "/${line}/,/^[0-9]/p" ${file_name} | grep "<")
            diff_file_name=$(sed -n "/${line}/,/^[0-9]/p" ${file_name} | grep -Po "<.*[a-z,A-Z].*.py" | sed "s|<||g")
            #diff_file_name=$(echo ${diff_lines} | grep -Po "<.*[a-z,A-Z].*.py" | sed "s|,||g)
            for diff_file in ${diff_file_name}
            do          
              diff_file=$(echo "${diff_file}" | sed 's/[ \t]*//g')
              file1=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $1}')
              miss1=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $3}')
              cover1=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $6}')
              branch1=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $4}')
              file2=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $1}')
              miss2=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $3}')
              cover2=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $6}')
              branch2=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po ">.*" | sed 's/>[ \t]*//g' | awk '{print $4}')
              # if branch coverage not change, not consider as regression
              [[ "${branch1}" == "${branch2}" ]] && continue
              echo """
              <tr><td>PR | BASE</td><td style=\"text-align:left\">${file1}</td>
                  <td style=\"text-align:left\">${miss1} | ${miss2}</td>
                  <td style=\"text-align:left\">${branch1} | ${branch2}</td>
                  <td style=\"text-align:left\">${cover1} | ${cover2}</td>
              </tr>""" >> ${output_file}
            done
        elif [[ $(echo $line | grep "[0-9]d[0-9]") ]] && [[ $(cat $file_name | grep -A 1 "$line" | grep "<") ]]; then
            diff_lines=$(sed -n "/${line}/,/^[0-9]/p" ${file_name} | grep "<")
            diff_file_name=$(sed -n "/${line}/,/^[0-9]/p" ${file_name} | grep -Po "<.*[a-z,A-Z].*.py" | sed "s|<||g")
            for diff_file in ${diff_file_name}
            do    
              diff_file=$(echo "${diff_file}" | sed 's/[ \t]*//g')
              file=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $1}')
              miss=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $3}')
              cover=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $6}')
              branch=$(cat $file_name | grep "${diff_file}" | grep -v ".*/${diff_file}" | grep -Po "<.*" | sed 's/<[ \t]*//g' | awk '{print $4}')
              echo """
              <tr><td>PR | BASE</td><td style=\"text-align:left\">${file} | NA</td>
                  <td style=\"text-align:left\">${miss} | NA</td>
                  <td style=\"text-align:left\">${branch} | NA</td>
                  <td style=\"text-align:left\">${cover} | NA</td>
              </tr>""" >> ${output_file}
            done
        fi
    done
    # generate table end
    echo """</table></div></body></html>""" >> ${output_file}

}

main
