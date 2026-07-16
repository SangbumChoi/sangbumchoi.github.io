---
layout: jarvis-content
title: "Research"
permalink: /publications/
eyebrow: "DANIEL OS / PUBLICATIONS"
dek: "Multimodal grounding, mobile vision, signal processing, and applied machine learning."
author_profile: false
---

{% include base_path %}

<div class="content-actions">
  <a class="signal-link" href="https://scholar.google.co.kr/citations?user=4klHsscAAAAJ&hl=ko" target="_blank" rel="noopener">Google Scholar ↗</a>
  <a class="signal-link" href="https://arxiv.org/a/choi_s_11.html" target="_blank" rel="noopener">arXiv author page ↗</a>
</div>

<div class="research-index">
{% for post in site.publications reversed %}
  <article class="research-entry">
    <div class="research-entry__meta">
      <span>{{ post.date | date: "%Y" }}</span>
      <span>{{ post.venue }}</span>
    </div>
    <h2><a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a></h2>
    {% if post.excerpt %}<p>{{ post.excerpt }}</p>{% endif %}
    <div class="research-entry__links">
      <a href="{{ base_path }}{{ post.url }}">Details</a>
      {% if post.paperurl and post.paperurl != '' %}<a href="{{ post.paperurl }}" target="_blank" rel="noopener">Paper ↗</a>{% endif %}
    </div>
  </article>
{% endfor %}
</div>
