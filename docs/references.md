# References

The papers below are the source of the methods implemented in this repository.
The PDFs are not redistributed here for copyright reasons; each entry links to
the publisher's page.

## Segmentation

**Statistical Region Merging** — the basis of `dermoseg.segmentation.region_merging`

> Celebi, M. E., Kingravi, H. A., Iyatomi, H., Aslandogan, Y. A., Stoecker, W. V.,
> Moss, R. H., Malters, J. M., Grichnik, J. M., Marghoob, A. A., Rabinovitz, H. S.,
> & Menzies, S. W. (2008). *Border detection in dermoscopy images using statistical
> region merging.* Skin Research and Technology, 14(3), 347–353.
> https://doi.org/10.1111/j.1600-0846.2008.00301.x

The underlying merging algorithm is Nock & Nielsen's Statistical Region Merging
(union–find over pixel pairs ordered by intensity difference).

**LBP Clustering** — the basis of `dermoseg.segmentation.lbp`

> Pereira, P. M. M., Fonseca-Pinto, R., Paiva, R. P., Assunção, P. A. A., Tavora, L. M. N.,
> Thomaz, L. A., & Faria, S. M. M. (2020). *Dermoscopic skin lesion image segmentation
> based on Local Binary Pattern Clustering: Comparative study.* Biomedical Signal
> Processing and Control, 59, 101924. (Open access, CC BY-NC-ND)

Source of the LBP P=8/R=1 pattern-subset binarisation, the `[L, Y, L]` pseudo-RGB
construction and the *pinkness* criterion `max(a*, 0) − min(b*, 0)`.

**Colour-channel thresholding** — background for `dermoseg.segmentation.otsu`

> Garnavi, R., Aldeen, M., Celebi, M. E., Bhuiyan, A., Dolianitis, C., & Varigos, G. (2009).
> *Skin Lesion Segmentation Using Color Channel Optimization and Clustering-based
> Histogram Thresholding.* International Journal of Biomedical and Biological
> Engineering, 3(12). https://waset.org/Publication/9437

> Zortea, M., Flores, E., & Scharcanski, J. (2017). *A simple weighted thresholding method
> for the segmentation of pigmented skin lesions in macroscopic images.*
> Pattern Recognition, 64, 92–104.

Source of the border-band skin estimate reused in the region-scoring step.

## Preprocessing

**Hair removal** — the basis of `dermoseg.preprocessing.remove_hair`

> Lee, T., Ng, V., Gallagher, R., Coldman, A., & McLean, D. (1997). *DullRazor: A software
> approach to hair removal from images.* Computers in Biology and Medicine, 27(6), 533–543.

Source of the grayscale-morphology hair detection (closing with directional
structuring elements, then inpainting of the detected pixels).

> Koehoorn, J., Sobiecki, A. C., Boda, D., Diaconeasa, A., Doshi, S., Paisey, S.,
> Jalba, A., & Telea, A. (2015). *Automated Digital Hair Removal by Threshold Decomposition
> and Morphological Analysis.* ISMM 2015, LNCS 9082, 15–26.
> https://doi.org/10.1007/978-3-319-18720-4_2

Reviewed but not implemented.

> Cavalcanti, P. G., Scharcanski, J., & Lopes, C. B. O. (2010). *Shading Attenuation in
> Human Skin Color Images.* ISVC 2010, LNCS 6453, 190–198.

Reviewed but not implemented.

## Project report

`research_paper.pdf` in this folder is the report written for the course.
