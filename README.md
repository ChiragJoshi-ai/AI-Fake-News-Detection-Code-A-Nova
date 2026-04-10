<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fake News Detector</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono&display=swap" rel="stylesheet">
<style>
  body {
    font-family: 'IBM Plex Sans', sans-serif;
    max-width: 680px;
    margin: 60px auto;
    padding: 0 24px 80px;
    color: #1c1c1c;
    background: #fff;
    font-size: 15px;
    line-height: 1.7;
  }

  h1 { font-size: 22px; font-weight: 500; margin-bottom: 6px; }
  h2 { font-size: 15px; font-weight: 500; margin: 36px 0 10px; border-bottom: 1px solid #e5e5e5; padding-bottom: 6px; }

  p { color: #333; margin-bottom: 12px; }

  .subtitle { color: #888; font-size: 14px; margin-bottom: 32px; }

  .tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    background: #f3f3f3;
    color: #555;
    padding: 2px 8px;
    border-radius: 3px;
    margin: 2px 2px 2px 0;
  }

  .step { display: flex; gap: 14px; margin-bottom: 14px; align-items: flex-start; }
  .step-n { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #aaa; padding-top: 3px; min-width: 20px; }
  .step-text { font-size: 14px; color: #333; }
  .step-text strong { font-weight: 500; color: #1c1c1c; }

  .signal { margin-bottom: 10px; }
  .signal-name { font-weight: 500; font-size: 14px; }
  .signal-desc { font-size: 13px; color: #666; }

  .note {
    background: #fafafa;
    border-left: 3px solid #ddd;
    padding: 10px 14px;
    font-size: 13px;
    color: #555;
    margin: 20px 0;
  }

  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  td { padding: 7px 10px; border-bottom: 1px solid #f0f0f0; color: #444; }
  td:first-child { font-family: 'IBM Plex Mono', monospace; color: #1c1c1c; width: 140px; }
</style>
</head>
<body>

<h1>AI Fake News Detector</h1>
<p class="subtitle">NLP classifier · FastAPI · scikit-learn</p>

<p>
  A machine learning system that reads a news article and tells you whether it's likely real or fake.
  It doesn't just look at keywords — it analyzes how the article is written, combining statistical
  text features with linguistic patterns that commonly show up in misinformation.
</p>

<h2>How it works</h2>
<p>
  When you paste an article, the text goes through a preprocessing step that strips out HTML,
  normalizes URLs and numbers, and lowercases everything. It then gets passed through two feature
  extractors at the same time.
</p>
<p>
  The first is a TF-IDF vectorizer that turns the text into 5,000 weighted word and phrase features,
  with bigrams so it can pick up two-word patterns like "deep state" or "breaking news".
  The second computes five hand-crafted signals from the raw text. Both sets of features get merged
  and fed into a logistic regression classifier that outputs a real or fake label with a confidence score.
</p>

<h2>Signals</h2>
<p>These are the five linguistic features the model computes on every article:</p>

<div class="signal"><span class="signal-name">Exclamation density</span> — <span class="signal-desc">how many exclamation marks appear relative to the length of the article. Fake articles use them a lot more.</span></div>
<div class="signal"><span class="signal-name">ALL-CAPS ratio</span> — <span class="signal-desc">proportion of words written in full capitals. A common tactic to push urgency and emotion.</span></div>
<div class="signal"><span class="signal-name">Lexical diversity</span> — <span class="signal-desc">ratio of unique words to total words. Real journalism tends to use a wider, more varied vocabulary.</span></div>
<div class="signal"><span class="signal-name">Sentence length variance</span> — <span class="signal-desc">how erratically sentence lengths jump around. Fabricated content often has very uneven structure.</span></div>
<div class="signal"><span class="signal-name">Red-flag phrases</span> — <span class="signal-desc">specific sensationalist keywords like BREAKING, BOMBSHELL, SHARE BEFORE DELETED, DEEP STATE, etc.</span></div>

<h2>Setup</h2>

<div class="step"><span class="step-n">01</span><span class="step-text"><strong>Install dependencies</strong> — use uv to install everything from requirements.txt</span></div>
<div class="step"><span class="step-n">02</span><span class="step-text"><strong>Train the model</strong> — run train.py, it saves the trained model as detector.pkl and prints accuracy</span></div>
<div class="step"><span class="step-n">03</span><span class="step-text"><strong>Start the API</strong> — launch app.py with uvicorn on port 8000</span></div>
<div class="step"><span class="step-n">04</span><span class="step-text"><strong>Open the UI</strong> — double-click index.html in your browser, the status dot turns green when the API is connected</span></div>

<div class="note">
  Train the model before starting the API — the server looks for detector.pkl on startup and will crash if it's missing.
</div>

<h2>Accuracy</h2>
<p>
  The default setup trains on 30 built-in examples and gets around 75% accuracy. That's expected for
  a dataset this small — it's enough to show the system working. To get to 95%+, swap in the
  ISOT Fake News Dataset from Kaggle (40,000+ articles). Just replace the dataset loader in train.py,
  everything else stays the same.
</p>

<h2>Tech stack</h2>
<table>
  <tr><td>Language</td><td>Python 3.10+</td></tr>
  <tr><td>ML</td><td>scikit-learn, scipy</td></tr>
  <tr><td>NLP</td><td>TF-IDF, n-gram extraction</td></tr>
  <tr><td>API</td><td>FastAPI + Uvicorn</td></tr>
  <tr><td>UI</td><td>HTML, CSS, JavaScript</td></tr>
  <tr><td>Packages</td><td>uv</td></tr>
</table>

<h2>Files</h2>
<table>
  <tr><td>train.py</td><td>trains the model and saves detector.pkl</td></tr>
  <tr><td>app.py</td><td>FastAPI inference server</td></tr>
  <tr><td>detector.pkl</td><td>saved model, generated after training</td></tr>
  <tr><td>index.html</td><td>browser UI</td></tr>
  <tr><td>requirements.txt</td><td>Python dependencies</td></tr>
</table>

</body>
</html>