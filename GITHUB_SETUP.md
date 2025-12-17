# GitHub Repository Setup Instructions

Follow these steps to push your project to GitHub and share it with your friend.

## Step 1: Create GitHub Repository

1. Go to [https://github.com/new](https://github.com/new)
2. **Repository name**: `hr-risk-analysis`
3. **Description**: "Enterprise-grade Employee Attrition Risk Prediction System with F500 corporate dashboard"
4. Choose **Public** (so your friend can access it) or **Private** (if you prefer)
5. **IMPORTANT**: Do NOT check "Initialize this repository with a README" (we already have one)
6. Do NOT add .gitignore or license (we already have them)
7. Click **"Create repository"**

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
cd /Users/tuankhailai/Downloads/Turnover-Predictor-main

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/hr-risk-analysis.git

# Rename branch to main (if needed)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

**Note**: You may be prompted for your GitHub username and password (or personal access token).

## Step 3: Verify Upload

1. Go to your repository page: `https://github.com/YOUR_USERNAME/hr-risk-analysis`
2. You should see all your files including:
   - `README.md`
   - `app.py`
   - `model/` folder with .pkl files
   - `employee_attrition_dataset_final.csv`
   - All other project files

## Step 4: Share with Your Friend

Send your friend this link:
```
https://github.com/YOUR_USERNAME/hr-risk-analysis
```

They can then follow the instructions in `SETUP.md` or `README.md` to clone and run the project.

## Authentication Note

If you encounter authentication issues when pushing:

1. **Use Personal Access Token** (recommended):
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate a new token with `repo` permissions
   - Use this token as your password when pushing

2. **Or use SSH** (alternative):
   - Set up SSH keys with GitHub
   - Use SSH URL: `git@github.com:YOUR_USERNAME/hr-risk-analysis.git`

## Troubleshooting

**Issue: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/hr-risk-analysis.git
```

**Issue: Authentication failed**
- Make sure you're using a Personal Access Token, not your GitHub password
- Or set up SSH keys for easier authentication

**Issue: "failed to push some refs"**
- Make sure you've created the repository on GitHub first
- Check that the repository name matches exactly

---

Once pushed, your friend can clone the repository and run the dashboard following the instructions in `SETUP.md`!

