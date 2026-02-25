final_chain= RunnableSequence(reportGenerationChain,branch_chain)
# OR#

LCEL_Pipe_final_chain = reportGenerationChain | branch_chain