import os, sys
import glob

def find_index_path(index_file):
    with open(index_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            pos = line.find('index.html" class="icon icon-home"')
            if pos<0:
                continue
            pos1 = line.rfind("\"", 0, pos)
            if pos1<0:
                return ""
            else:
                return "../" + line[pos1+1: pos]
    return "ignore"

def update_version_link(version, folder_name, index_file):
    index_buf = ""
    index_path = find_index_path(index_file)
    if index_path=='ignore':
        return

    with open(index_file, "r") as f:
        index_buf = f.read()
        key_str='  <div class="version">\n                {}\n              </div>'.format(version)
        version_list = '''<div class="version">
              <a href="{}versions.html">{}▼</a>
              <p>Click link above to switch version</p>
            </div>'''.format(index_path, folder_name)
        #print(index_buf.find(key_str))
        index_buf = index_buf.replace(key_str, version_list)
        #print(index_buf)

    with open(index_file, "w") as f:
        f.write(index_buf)


def main(folder, version):
    folder_name=os.path.basename(folder)
    for index_file in glob.glob('{}/**/*.html'.format(folder),recursive = True):
        update_version_link(version, folder_name, index_file)


def help(me):
    print("python {} html_folder version".format(me))

if __name__=="__main__":
    if len(sys.argv)<3:
        help(sys.argv[0])
        sys.exit(1)

    folder = sys.argv[1]
    version = sys.argv[2]
    main(folder, version)
