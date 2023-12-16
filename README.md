# CRCS
 
 An automatic image processing pipeline for hormone level analysis of cushing’s disease, which aims to explore the relationship between the expression level of adreno-cortico-tropic-hormone (ACTH) in normal cell tissues adjacent to tumor cells and the postoperative recurrence rate of patients. 
 
 CRCS mainly consists of image-level clustering, cluster-level multi-modal image registration, patch-level image classification and pixel-level image segmentation on the whole slide imaging (WSI). 
 
 
 Automaic image registration is the first step of the whole registration process. There are some cases where the impurity area is larger than the effective content, which cannot be successfully registered in theory. For the image group with failed automatical registeration, the pathologists are required to register them manually. For this purpose, we have developed a graphical user interface (GUI) in MATLAB Uiaxes for interactive translation and rotation operations. The user starts to control the translation movement of the moving image in the form of dragging after clicking the image, and stops moving after clicking it again. The moving image remains the same track as the mouse in real time. And the user can adjust the rotation angle of the moving image by the angleslider. In this way, a user-friendly registeration fuction block is formed with human-in-the-loop.
 
# samples

 Three samples from the CRCS clinical dataset are provided for program reproduction purposes only.
 
 When running the program, please locate the "Browse" button to "sample_block".

 The data used in this article involves medical ethics issues, so it cannot be directly published. Researchers who are interested in this research can communicate with us. We can share data with a certain permission. 
 
# sample block 
 If the original clinical image is too large to upload or download. Readers can find the phased results of hierarchical clustering here, which can be used to complete the remaining program  reproduction. 

# Ethics statement
 The Ethics Committee of Ruijin Hospital, Shanghai Jiaotong University School of Medicine approved the study protocol.