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
