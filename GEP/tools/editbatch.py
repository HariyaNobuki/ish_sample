import os
import sys

problemname = ["Nguyen7"]

def CreateBatfile(wfilepath ,StrData=[]):

    wfile=open(wfilepath + "/" + wfilename + ".bat",'w')
    #wfile.write('cd '+s0+str(ScriptPath()[2])+s0+'\n')
    #wfile.write('echo off'+'\n')
    wfile.write('python '+s0+str(sys.argv[0])+s0+' '+' '.join(StrData)+'\n')

    #writepath=WriteFilePath(Pathname='ReadWRite_Data')
    #s1="rem EXPLORER "
    #s2=writepath
    #s=s1+s0+s2+s0
    #wfile.write(s+' \n')
    wfile.write('pause \n')
    wfile.flush()
    wfile.close


def WriteFilePath(Pathname='ReadWrite_Data'):     
    writepath=os.path.join(ScriptPath()[2],Pathname)
    if not os.path.isdir(writepath):
        os.mkdir(writepath)
    return writepath

def ScriptPath():
    # ■## ScriptPath():
    #ScriptPath()[0]:basename ScriptPath()[1]:pathname ScriptPath()[2]:fulpath  
    basename=os.path.basename((sys.argv[0]))
    pathname=os.path.abspath(os.path.dirname(sys.argv[0]))
    fullpath=os.path.join(pathname,basename)
    return (basename,fullpath,pathname)


if __name__  == '__main__':
    c_path = os.getcwd()
    _ex_path = c_path+"/_ex_batch"
    wfilepath = _ex_path + "/" + problemname[0]
    os.makedirs(wfilepath , exist_ok=True)

    CreateBatfile(wfilepath,StrData=[])
    #