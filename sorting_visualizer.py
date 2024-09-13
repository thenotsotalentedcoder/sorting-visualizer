import streamlit as st
import random
import time
import pandas as pd
import altair as alt
from typing import List, Tuple

def generate_starting_list(n: int, min_val: int, max_val: int) -> List[int]:
    return [random.randint(min_val, max_val) for _ in range(n)]

def bubble_sort(arr: List[int], ascending: bool = True) -> List[Tuple[List[int], List[int]]]:
    steps = []
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if (arr[j] > arr[j + 1] and ascending) or (arr[j] < arr[j + 1] and not ascending):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                steps.append((arr.copy(), [j, j + 1]))
    return steps

def insertion_sort(arr: List[int], ascending: bool = True) -> List[Tuple[List[int], List[int]]]:
    steps = []
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while (j >= 0 and key < arr[j] and ascending) or (j >= 0 and key > arr[j] and not ascending):
            arr[j + 1] = arr[j]
            j -= 1
            steps.append((arr.copy(), [j + 1, i]))
        arr[j + 1] = key
        steps.append((arr.copy(), [j + 1, i]))
    return steps

def merge_sort(arr: List[int], ascending: bool = True) -> List[Tuple[List[int], List[int]]]:
    steps = []

    def merge(left: List[int], right: List[int]) -> List[int]:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if (left[i] <= right[j] and ascending) or (left[i] >= right[j] and not ascending):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def merge_sort_recursive(arr: List[int], start: int, end: int):
        if end - start > 1:
            mid = (start + end) // 2
            merge_sort_recursive(arr, start, mid)
            merge_sort_recursive(arr, mid, end)
            merged = merge(arr[start:mid], arr[mid:end])
            arr[start:end] = merged
            steps.append((arr.copy(), list(range(start, end))))

    merge_sort_recursive(arr, 0, len(arr))
    return steps

def quick_sort(arr: List[int], ascending: bool = True) -> List[Tuple[List[int], List[int]]]:
    steps = []

    def partition(low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if (arr[j] <= pivot and ascending) or (arr[j] >= pivot and not ascending):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                steps.append((arr.copy(), [i, j]))
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append((arr.copy(), [i + 1, high]))
        return i + 1

    def quick_sort_recursive(low: int, high: int):
        if low < high:
            pi = partition(low, high)
            quick_sort_recursive(low, pi - 1)
            quick_sort_recursive(pi + 1, high)

    quick_sort_recursive(0, len(arr) - 1)
    return steps

def visualize_sort(arr: List[int], steps: List[Tuple[List[int], List[int]]]):
    chart = st.empty()
    progress_bar = st.progress(0)
    
    for i, (step, highlight) in enumerate(steps):
        df = pd.DataFrame({'index': range(len(step)), 'value': step})
        
        bars = alt.Chart(df).mark_bar().encode(
            x='index:O',
            y='value:Q',
            color=alt.condition(
                alt.FieldOneOfPredicate(field='index', oneOf=highlight),
                alt.value('red'),
                alt.value('steelblue')
            )
        ).properties(width=600, height=400)
        
        chart.altair_chart(bars)
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.1)  # Adjust speed of visualization

def main():
    st.set_page_config(page_title="Sorting Algorithm Visualizer", page_icon="📊", layout="wide")
    st.title("Sorting Algorithm Visualizer")

    col1, col2 = st.columns(2)

    with col1:
        n = st.slider("Number of elements", min_value=10, max_value=100, value=50)
        min_val = st.number_input("Minimum value", value=0)
        max_val = st.number_input("Maximum value", value=100)

    with col2:
        algorithm = st.selectbox("Select sorting algorithm", 
                                 ["Bubble Sort", "Insertion Sort", "Merge Sort", "Quick Sort"])
        ascending = st.checkbox("Sort in ascending order", value=True)

    if 'original_arr' not in st.session_state or st.button("Generate New Array"):
        st.session_state.original_arr = generate_starting_list(n, min_val, max_val)
        st.session_state.arr = st.session_state.original_arr.copy()
        st.session_state.sorted = False
        st.session_state.sorting_time = None

    if st.button("Revert to Original Array"):
        st.session_state.arr = st.session_state.original_arr.copy()
        st.session_state.sorted = False
        st.session_state.sorting_time = None

    if st.button("Sort"):
        if not st.session_state.sorted:
            arr = st.session_state.arr.copy()
            
            if algorithm == "Bubble Sort":
                start_time = time.perf_counter()
                steps = bubble_sort(arr, ascending)
                end_time = time.perf_counter()
            elif algorithm == "Insertion Sort":
                start_time = time.perf_counter()
                steps = insertion_sort(arr, ascending)
                end_time = time.perf_counter()
            elif algorithm == "Merge Sort":
                start_time = time.perf_counter()
                steps = merge_sort(arr, ascending)
                end_time = time.perf_counter()
            else:  # Quick Sort
                start_time = time.perf_counter()
                steps = quick_sort(arr, ascending)
                end_time = time.perf_counter()

            st.session_state.sorting_time = end_time - start_time

            visualize_sort(st.session_state.arr, steps)
            st.session_state.sorted = True
            st.session_state.arr = arr

    if st.session_state.sorted and st.session_state.sorting_time is not None:
        sorting_time_ms = st.session_state.sorting_time * 1000  # Convert to milliseconds
        st.success(f"Sorting completed in {sorting_time_ms:.3f} ms")

    # Display the current array
    st.subheader("Current Array")
    st.bar_chart(st.session_state.arr)

if __name__ == "__main__":
    main()

