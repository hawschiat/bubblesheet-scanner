import cv2 as cv
import numpy as np
from imutils import contours, grab_contours
from pathlib import Path

"""
Bubble sheet scanner script, based on the implementation at:
https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
"""


def get_image_as_mat(path):
    path = Path(path)
    # Read the file content
    with open(path, 'rb') as f:
        content = bytearray(f.read())
        mat = np.asarray(content, dtype=np.uint8)
        src = cv.imdecode(mat, flags=cv.IMREAD_GRAYSCALE)
        if src is None:
            raise ValueError('The file seems to be of invalid type or encoding!')

        # ---Sharpening filter----
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        src = cv.filter2D(src, -1, kernel)

        return src


def get_bin_threshold(src, log_path: str = ''):
    if src is None:
        raise ValueError('Provided data is invalid!')
    # apply Otsu's thresholding method to binarize the warped piece of paper
    thresh = cv.threshold(src, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    # Denoise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))
    thresh = cv.erode(thresh, kernel, iterations=2)

    # ---Sharpening filter----
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    thresh = cv.filter2D(thresh, -1, kernel)

    # Dilate image
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    thresh = cv.dilate(thresh, kernel, iterations=1)

    if log_path:
        p = Path(log_path)
        new_path = f"{p.parent.absolute()}/{p.stem}_thresh.jpg"
        cv.imwrite(new_path, thresh)

    return thresh


def identify_id(src, id_length: int = 10, log_path: str = ''):
    if src is None:
        raise ValueError('Provided data is invalid!')
    thresh = get_bin_threshold(src, log_path=log_path)
    # find contours in the thresholded image, then initialize the list of possible contours
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    box_cnts = []

    # global width and height
    height, width = thresh.shape
    min_box_width = width/3

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a region for id, the box
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= min_box_width and 1.1 <= ar <= 3:
            box_cnts.append(c)
            break

    if len(box_cnts) == 0:
        raise ValueError("Could not identify the bubble region for student ID!")

    # construct a mask that reveals only the current region to identify the bubbles
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv.boundingRect(box_cnts[0])
    cv.rectangle(mask, (x+3, y+3), (x+w-6, y+h-6), (255, 255, 255), -1)
    # cv.drawContours(mask, box_cnts, -1, 255, -1)

    # apply the mask to the thresholded image, then scan for bubbles in the region
    mask = cv.bitwise_and(thresh, thresh, mask=mask)
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnts = grab_contours(cnts)
    bubble_cnts = []

    # if log_path:
    #     rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    #     p = Path(log_path)
    #     new_path = f"{p.parent.absolute()}/{p.stem}_id_mask.jpg"
    #     cv.imwrite(new_path, rgb)

    min_bubble_size = width / 50

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= min_bubble_size and h >= min_bubble_size and 0.9 <= ar <= 1.1:
            bubble_cnts.append(c)

    if len(bubble_cnts) == 0:
        raise ValueError("Could not find the bubbles!")

    # assert(len(bubble_cnts) == 100)

    # sort the bubble contours left-to-right
    bubble_cnts = contours.sort_contours(bubble_cnts)[0]

    bubble_matrix = []
    for (b, i) in enumerate(np.arange(0, len(bubble_cnts), 10)):
        # sort the contours for this row top-to-bottom, then append to the list
        cnts = (contours.sort_contours(bubble_cnts[i:i + 10], method="top-to-bottom")[0])[::1]
        bubble_matrix.append(cnts)

    if log_path:
        rgb = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
        # Give different colors to different columns
        with_contour = None
        for c in bubble_matrix:
            color = list(np.random.random(size=3) * 256)
            with_contour = cv.drawContours(rgb, c, -1, color, 2)
        p = Path(log_path)
        new_path = f"{p.parent.absolute()}/{p.stem}_id_area.jpg"
        cv.imwrite(new_path, with_contour)

    # Start detecting ID
    detected_id = ""
    for cnts in bubble_matrix:
        # initialize bubble state
        bubbled = None
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv.drawContours(mask, [c], -1, 255, -1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv.bitwise_and(thresh, thresh, mask=mask)
            total = cv.countNonZero(mask)

            # retrive the bound of this bubble
            bound = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            bound = grab_contours(bound)

            # if the current total has a total of non-zero pixels larger than 2/3 of the contour size,
            # then we are examining the currently bubbled-in answer
            if total > (2 * cv.contourArea(bound[0]) / 3):
                bubbled = (total, j)

        if bubbled is None:
            raise ValueError("Missing digit in the ID field!")

        detected_id += f"{bubbled[1]}"

    return detected_id


def identify_bubbles(src, choice_count: int = 4, log_path: str = ''):
    if src is None:
        raise ValueError('Provided data is invalid!')
    thresh = get_bin_threshold(src, log_path=log_path)
    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    question_cnts = []
    box_cnts = []

    # global width and height
    height, width = thresh.shape
    min_box_width = width / 2
    min_bubble_size = width / 50

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv.boundingRect(c)
        ar = w / float(h)

        # find the boundaries that mark the start and end of the bubble region
        if w >= min_box_width and ar >= 2:
            box_cnts.append(c)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= min_bubble_size and h >= min_bubble_size and 0.9 <= ar <= 1.1:
            question_cnts.append(c)

    # sort the question contours top-to-bottom, then initialize
    # the total number of correct answers
    question_cnts, question_bounds = contours.sort_contours(question_cnts, method="top-to-bottom")
    question_bounds = np.asarray(question_bounds)
    # sort the box contours top-to-bottom to find out the closest boundaries for the bubble region
    box_cnts, box_bounds = contours.sort_contours(box_cnts, method="top-to-bottom")

    if len(box_cnts) > 2:
        box_bounds = list(box_bounds)[-2:]
        box_cnts = box_cnts[-2:]

    # filter out any potential bubbles that are outside of the boundaries
    filtered_question_cnts = []
    filtered_question_bounds = []
    top_bound = box_bounds[0][1]
    bottom_bound = box_bounds[1][1]
    # using an index to track where we are at, as using plain enumerate messes up the loop (this is a 2d array)
    index = 0
    for c in question_cnts:
        y = question_bounds[index][1]
        if top_bound <= y <= bottom_bound:
            filtered_question_cnts.append(c)
            filtered_question_bounds.append(question_bounds[index])
        index += 1

    # replace original lists with the filtered ones
    question_cnts = filtered_question_cnts
    question_bounds = filtered_question_bounds

    if log_path:
        rgb = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
        with_contour = cv.drawContours(rgb, question_cnts, -1, (0, 255, 0), 2)
        with_contour = cv.drawContours(with_contour, box_cnts, -1, (0, 0, 255), 2)
        p = Path(log_path)
        new_path = f"{p.parent.absolute()}/{p.stem}_marked_choices.jpg"
        cv.imwrite(new_path, with_contour)

    # store the bubbled choices in a list
    bubbled_choices = []
    current_row = []
    current_y = question_bounds[0][1]

    # loop over the question in batches of the defined number of choice
    for (q, i) in enumerate(np.arange(0, len(question_cnts), choice_count)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts = contours.sort_contours(question_cnts[i:i + choice_count])[0]
        bubbled = []

        # if the current bubbles are lower than the current minimum,
        # then we append the current row to the master list
        y = max(list(map(lambda b: b[1], question_bounds[i:i + choice_count])))
        if y > (current_y + height / 100):
            bubbled_choices.append(current_row)
            current_y = y
            current_row = []

        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv.drawContours(mask, [c], -1, 255, -1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv.bitwise_and(thresh, thresh, mask=mask)
            total = cv.countNonZero(mask)

            # retrive the bound of this bubble
            bound = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            bound = grab_contours(bound)

            # if the current total has a total of non-zero pixels larger than 2/3 of the contour size,
            # then we are examining the currently bubbled-in answer
            if total > (2 * cv.contourArea(bound[0]) / 3):
                bubbled.append((total, j))

        # put the selected choice(s) into the master list
        if len(bubbled) == 0:
            current_row.append([-1])
        else:
            current_row.append(list(map(lambda b: b[1], bubbled)))

    # Append the last row to the list
    if len(current_row) > 0:
        bubbled_choices.append(current_row)

    # Transpose the matrix to get the right order
    transpose = [[bubbled_choices[j][i] for j in range(len(bubbled_choices))] for i in range(len(bubbled_choices[0]))]
    return [arr for subarr in transpose for arr in subarr]


def main(paths):
    for path in paths:
        src = get_image_as_mat(path)
        student_id = identify_id(src, log_path=path)
        print(student_id)
        selected = identify_bubbles(src, log_path=path)
        print(selected)


if __name__ == '__main__':
    main(['tests/test_filled-corrected.jpg'])
