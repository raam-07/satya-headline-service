from headline_pipeline import validate_formatting, post_process_headline, should_skip_article

def test_should_skip_article():
    # Length skip rule (> 15000 chars)
    assert should_skip_article("Original Title", "a" * 15001) == True
    assert should_skip_article("Original Title", "a" * 15000) == False
    
    # Title skip rules
    assert should_skip_article("Live Updates: election results", "content") == True
    assert should_skip_article("election results explained in detail", "content") == True
    assert should_skip_article("election guide from A to Z", "content") == True
    assert should_skip_article("Title | Segment 2 | Segment 3", "content") == True
    
    # Opinion skip rules
    assert should_skip_article("Normal Title", "Opinion: this is a column") == True
    assert should_skip_article("Opinion | Normal Title", "content") == True
    assert should_skip_article("Normal Title | Comment", "content") == True
    assert should_skip_article("Editorial: Normal Title", "content") == True

    # Regular title and content should not skip
    assert should_skip_article("Normal Headline Title", "Short content") == False

def test_post_process_headline():
    # Trailing periods
    assert post_process_headline("This is a headline.") == "This is a headline"
    
    # Wrapping quotes and asterisks
    assert post_process_headline('"This is a headline"') == "This is a headline"
    assert post_process_headline("'This is a headline'") == "This is a headline"
    assert post_process_headline('""This is a headline""') == "This is a headline"
    assert post_process_headline("*This is a headline*") == "This is a headline"
    assert post_process_headline("**This is a headline**") == "This is a headline"
    assert post_process_headline('"*This is a headline*"') == "This is a headline"
    
    # Collapse whitespace
    assert post_process_headline("This   is   a   headline") == "This is a headline"
    
    # Sentence case first word, keep rest of casing
    assert post_process_headline("this is a headline about Modi") == "This is a headline about Modi"
    assert post_process_headline("BJP wins election") == "BJP wins election"

def test_validate_formatting():
    # Valid headline
    valid, reason = validate_formatting("BJP wins elections in landmark victory")
    assert valid is True
    assert reason is None
    
    # Empty headline
    valid, reason = validate_formatting("")
    assert valid is False
    assert reason == "empty headline"
    
    valid, reason = validate_formatting("   ")
    assert valid is False
    assert reason == "empty headline"
    
    # Too short headline (< 3 words)
    valid, reason = validate_formatting("Bribe case")
    assert valid is False
    assert reason == "too short"
    
    # Too long headline (> 14 words)
    valid, reason = validate_formatting("This is a very long headline that exceeds the maximum limit of fourteen words and therefore must fail validation")
    assert valid is False
    assert "exceeds 14 words" in reason

    # Dangling word rejection
    valid, reason = validate_formatting("Minister has tightened rules on baby")
    assert valid is False
    assert "dangling ending word" in reason

    valid, reason = validate_formatting("Minister for Industries IT and AI P.K")
    assert valid is False
    assert "dangling ending word" in reason

    # Allow valid endings with digits
    valid, reason = validate_formatting("ISRO launches Chandrayaan 3")
    assert valid is True
    assert reason is None

    valid, reason = validate_formatting("Cabinet approves semiconductor mission phase 2")
    assert valid is True
    assert reason is None

def test_fallback_from_summary():
    from headline_pipeline import fallback_from_summary

    # Weekday date + initials preservation
    s1 = "On Wednesday, September 30, Minister for Industries, IT and AI P.K. Kunhalikutty announced the new policy."
    h1 = fallback_from_summary(s1)
    assert "P.K. Kunhalikutty" in h1 or "Kunhalikutty" in h1
    assert not h1.endswith("P.K")
    assert not h1.startswith("On Wednesday")

    # Appositive clause simplification without chopping on 'baby'
    s2 = "Tukaram Mundhe, leading the Maharashtra Food and Drug Administration, has tightened rules on baby foods."
    h2 = fallback_from_summary(s2)
    assert not h2.endswith("baby")
    assert "baby foods" in h2
    assert "Tukaram Mundhe" in h2

def test_prompt_templates():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(base_dir, "prompts")
    for fname in ["headline.txt", "headline_safe.txt", "critic.txt"]:
        p = os.path.join(prompt_dir, fname)
        assert os.path.exists(p), f"Missing prompt file: {fname}"
        content = open(p).read()
        assert "<start_of_turn>user" in content, f"Missing <start_of_turn>user in {fname}"
        assert "<end_of_turn>" in content, f"Missing <end_of_turn> in {fname}"
        assert "<start_of_turn>model" in content, f"Missing <start_of_turn>model in {fname}"
        assert "<|im_start|>" not in content, f"Stray ChatML token <|im_start|> in {fname}"
        assert "<|im_end|>" not in content, f"Stray ChatML token <|im_end|> in {fname}"
        if fname == "critic.txt":
            formatted = content.format(body_snippet="Sample Body", headline="Sample Headline")
            assert "Sample Body" in formatted and "Sample Headline" in formatted
        else:
            formatted = content.format(body_snippet="Sample Body")
            assert "Sample Body" in formatted


