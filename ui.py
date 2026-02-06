import gradio as gr
import requests
from database import get_all_results, get_disagreements_only, get_stats

API_URL = "http://localhost:8000/analyze-all"

def analyze_prompt(prompt):
    """Call the Chorus API and format results for display."""
    if not prompt.strip():
        return "Please enter a prompt to analyze."
    
    try:
        response = requests.post(API_URL, json={'content': prompt})
        result = response.json()
        
        # Format the output
        output = f"""
ANALYSIS RESULTS
================

Prompt: {prompt}

INDIVIDUAL MODEL RESULTS
------------------------

Claude Sonnet 4: {'SAFE' if result['claude']['safe'] else 'UNSAFE'}
Reasoning: {result['claude']['reasoning'][:150]}...

GPT-5.2: {'SAFE' if result['gpt5']['safe'] else 'UNSAFE'}
Reasoning: {result['gpt5']['reasoning'][:150]}...

Llama 3.1 70B: {'SAFE' if result['llama']['safe'] else 'UNSAFE'}
Reasoning: {result['llama']['reasoning'][:150]}...

CONSENSUS VERDICT
-----------------

Decision: {result['consensus']['verdict']}
Confidence: {result['consensus']['confidence']}
Flagged by: {', '.join(result['consensus']['flagged_by']) if result['consensus']['flagged_by'] else 'None'}

Result saved to database.
"""
        return output
        
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to API. Make sure server is running with: uvicorn main:app --reload"
    except Exception as e:
        return f"ERROR: {str(e)}"

def view_stats():
    """Show database statistics."""
    stats = get_stats()
    
    output = f"""
DATABASE STATISTICS
===================

Total Tests Run: {stats['total_tests']}

Breakdown by Verdict:
"""
    for verdict, count in stats['by_verdict'].items():
        output += f"  {verdict}: {count}\n"
    
    return output

def view_disagreements():
    """Show all disagreement cases."""
    disagreements = get_disagreements_only()
    
    if not disagreements:
        return "No disagreements found yet."
    
    output = f"DISAGREEMENT CASES ({len(disagreements)} total)\n"
    output += "=" * 50 + "\n\n"
    
    for row in disagreements:
        output += f"ID {row[0]}: {row[1][:80]}...\n"
        output += f"  Flagged by: {row[3]}\n"
        output += f"  Time: {row[4]}\n\n"
    
    return output

# Create Gradio interface
with gr.Blocks(title="Chorus - Multi-Model Safety Verification") as demo:
    gr.Markdown("# Chorus - Multi-Model AI Safety Verification")
    gr.Markdown("Compare how Claude, GPT-5, and Llama evaluate content safety")
    
    with gr.Tab("Analyze Prompt"):
        prompt_input = gr.Textbox(
            label="Enter prompt to analyze",
            placeholder="Type any prompt here...",
            lines=3
        )
        analyze_btn = gr.Button("Analyze with All Models", variant="primary")
        output = gr.Markdown()
        
        analyze_btn.click(fn=analyze_prompt, inputs=prompt_input, outputs=output)
    
    with gr.Tab("Database Stats"):
        stats_btn = gr.Button("Refresh Statistics")
        stats_output = gr.Markdown()
        stats_btn.click(fn=view_stats, outputs=stats_output)
    
    with gr.Tab("View Disagreements"):
        disagreements_btn = gr.Button("Show All Disagreements")
        disagreements_output = gr.Markdown()
        disagreements_btn.click(fn=view_disagreements, outputs=disagreements_output)

if __name__ == "__main__":
    demo.launch(share=False)