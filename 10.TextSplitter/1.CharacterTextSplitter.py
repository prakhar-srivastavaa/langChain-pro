from langchain_text_splitters import CharacterTextSplitter

text = '''Hey ﻿Prakhar Srivastava﻿
I'm reaching out from the Talent Team at TestMu AI. As we expand our Customer Engineering team ,we are looking for a few sharp, technically-minded interns to join us as Customer Engineer Intern. Since you’re already in our talent pool, we wanted to reach out to you first!
This isn’t a "sit on the sidelines" kind of internship. You’ll be diving deep into AI-native Quality Engineering, troubleshooting automation scripts (Selenium, Playwright, Appium), and working directly with global developers to solve complex technical challenges.
Please check this PDF for a detailed JD!
Our Selection Process
We like to keep things moving quickly! Here is how we’ll get started:
Please fill out this Google Form
Our HR team will review your profile and reach out to you if there's a match.
We are hosting a dedicated hiring drive this Friday, February 27th at our Noida office.
What the Friday Drive looks like:
Written Coding Test: A chance to show us your DSA fundamentals.
Group Discussion: We want to see how you collaborate and think on your feet.
Final Round: A conversation with our VP of Customer Engineering.
Why join us?
If you love solving puzzles, understanding how large-scale systems work, and want to gain hands-on experience with cloud-scale infrastructure, you'll fit right in. We’re a fast-paced, high-growth global B2B SaaS company, and we’re looking for people who are ready to take ownership and grow with us.
Looking forward to have you onboarded with us!'''

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_text(text)

print(result)