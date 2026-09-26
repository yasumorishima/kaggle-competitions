# Path check for the enveda-casmi26 submit workflow: copy the sample submission.
# Scores ~0 on purpose; it only proves push -> run -> submit works end to end.
import glob
import shutil

src = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
print("found:", src)
shutil.copy(src[0], "submission.csv")
print(open("submission.csv").read()[:200])
