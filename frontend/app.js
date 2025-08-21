// PaperWhisperer Frontend Application
class PaperWhispererApp {
    constructor() {
        this.apiBase = 'http://localhost:8000/api';
        this.currentPapers = [];
        this.currentPaper = null;
        this.currentPage = 1;
        this.totalPages = 1;
        this.isLoading = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupTabs();
        this.loadExistingPapers();
    }

    setupEventListeners() {
        // Search form
        document.getElementById('searchForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleDomainSearch();
        });

        // URL form
        document.getElementById('urlForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleUrlImport();
        });

        // Clear data button
        document.getElementById('clearDataBtn').addEventListener('click', () => {
            this.clearData();
        });

        // Chat button
        document.getElementById('chatBtn').addEventListener('click', () => {
            this.openChatModal();
        });

        // Modal page navigation
        document.getElementById('prevPageBtn').addEventListener('click', () => {
            this.navigatePage(-1);
        });

        document.getElementById('nextPageBtn').addEventListener('click', () => {
            this.navigatePage(1);
        });

        document.getElementById('pageSelector').addEventListener('change', (e) => {
            this.goToPage(parseInt(e.target.value));
        });

        // Analysis button
        document.getElementById('analyzePageBtn').addEventListener('click', () => {
            this.analyzePage();
        });
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Remove active class from all tabs
                tabButtons.forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Add active class to clicked tab
                button.classList.add('active');

                // Hide all tab contents
                tabContents.forEach(content => content.classList.add('hidden'));

                // Show corresponding content
                const tabId = button.id.replace('Tab', 'Content');
                document.getElementById(tabId).classList.remove('hidden');
            });
        });
    }

    async handleDomainSearch() {
        if (this.isLoading) return;

        const domain = document.getElementById('domain').value;
        const maxResults = parseInt(document.getElementById('maxResults').value);
        const clearExisting = document.getElementById('clearExisting').checked;

        // Add spinner to search button
        const searchButton = document.querySelector('#searchForm button[type="submit"]');
        this.addButtonSpinner(searchButton, 'Searching...');

        this.showLoading('Initializing neural search...');

        try {
            const response = await fetch(`${this.apiBase}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    domain,
                    max_results: maxResults,
                    clear_existing: clearExisting
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            this.handleSearchProgress(data);
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showError('Search failed. Please try again.');
        } finally {
            this.hideLoading();
            this.removeButtonSpinner(searchButton, 'Launch Neural Search');
        }
    }

    async handleUrlImport() {
        if (this.isLoading) return;

        const paperUrl = document.getElementById('paperUrl').value;
        const clearExisting = document.getElementById('clearExistingUrl').checked;

        if (!paperUrl) {
            this.showError('Please enter a valid ArXiv URL');
            return;
        }

        // Add spinner to import button
        const importButton = document.querySelector('#urlForm button[type="submit"]');
        this.addButtonSpinner(importButton, '📡 Importing...');

        this.showLoading('Importing paper...');

        try {
            const response = await fetch(`${this.apiBase}/import`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: paperUrl,
                    clear_existing: clearExisting
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.currentPapers = result.papers || [];
                this.displayPapers();
                this.showSuccess('Paper imported successfully!');
            } else {
                throw new Error(result.error || 'Import failed');
            }
        } catch (error) {
            console.error('Import error:', error);
            this.showError('Import failed. Please check the URL and try again.');
        } finally {
            this.hideLoading();
            this.removeButtonSpinner(importButton, '📡 Import Paper');
        }
    }

    handleSearchProgress(data) {
        if (data.type === 'progress') {
            this.updateProgress(data.current, data.total, data.message);
        } else if (data.type === 'paper') {
            this.addPaper(data.paper);
        } else if (data.type === 'complete') {
            this.currentPapers = data.papers || [];
            this.displayPapers();
            this.showSuccess(`Search completed! Found ${this.currentPapers.length} papers.`);
        } else if (data.type === 'error') {
            this.showError(data.message || 'Search failed');
        }
    }

    updateProgress(current, total, message) {
        const percentage = total > 0 ? (current / total) * 100 : 0;
        
        document.getElementById('loadingMessage').textContent = message;
        document.getElementById('progressBar').style.width = `${percentage}%`;
        document.getElementById('progressText').textContent = `${current}/${total} (${percentage.toFixed(1)}%)`;
    }

    addPaper(paper) {
        this.currentPapers.push(paper);
        this.displayPapers();
    }

    displayPapers() {
        const container = document.getElementById('papersContainer');
        const grid = document.getElementById('papersGrid');
        const paperCount = document.getElementById('paperCount');

        if (this.currentPapers.length === 0) {
            container.classList.add('hidden');
            return;
        }

        container.classList.remove('hidden');
        grid.innerHTML = '';

        // Update paper count
        if (paperCount) {
            paperCount.textContent = `${this.currentPapers.length} ${this.currentPapers.length === 1 ? 'paper' : 'papers'}`;
        }

        this.currentPapers.forEach((paper, index) => {
            const paperCard = this.createPaperCard(paper, index);
            grid.appendChild(paperCard);
        });
    }

    createPaperCard(paper, index) {
        const card = document.createElement('div');
        card.className = 'card paper-card overflow-hidden';

        const authorsText = paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ` +${paper.authors.length - 3} more` : '');
        const publishedDate = paper.published_date ? paper.published_date.slice(0, 10) : 'N/A';
        const categoriesText = paper.categories ? paper.categories.slice(0, 2).join(', ') : 'Research';
        const statusClass = paper.downloaded ? 'badge-success' : 'badge-processing';
        const statusText = paper.downloaded ? 'Ready' : 'Processing';
        const statusIcon = paper.downloaded ? 
            '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path></svg>' :
            '<div class="small-spinner"></div>';

        card.innerHTML = `
            <div class="p-6 bg-white">
                <div class="flex items-start justify-between mb-4">
                    <div class="flex-1 pr-4">
                        <h3 class="text-lg font-bold text-gray-800 leading-tight mb-2 line-clamp-2 font-source">
                            ${paper.title}
                        </h3>
                        <p class="text-sm text-gray-600 mb-3">
                            <span class="font-medium text-blue-600">Authors:</span> ${authorsText}
                        </p>
                    </div>
                    
                    <div class="${statusClass} px-3 py-2 rounded-lg text-white text-xs font-bold flex items-center space-x-2">
                        ${statusIcon}
                        <span>${statusText}</span>
                    </div>
                </div>
                
                <div class="grid grid-cols-3 gap-4 mb-6 text-xs">
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-3">
                        <div class="font-bold text-gray-500 uppercase tracking-wide mb-1">Published</div>
                        <div class="text-gray-800 font-mono">${publishedDate}</div>
                    </div>
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-3">
                        <div class="font-bold text-gray-500 uppercase tracking-wide mb-1">Categories</div>
                        <div class="text-gray-800">${categoriesText}</div>
                    </div>
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-3">
                        <div class="font-bold text-gray-500 uppercase tracking-wide mb-1">ArXiv ID</div>
                        <div class="text-gray-800 font-mono">${paper.arxiv_id || 'N/A'}</div>
                    </div>
                </div>
                
                <div class="flex space-x-3">
                    <button onclick="paperWhispererApp.openPaperPreview(${index})"
                            class="flex-1 btn-primary py-2.5 px-4 rounded-lg font-medium text-sm ${!paper.downloaded ? 'opacity-50 cursor-not-allowed' : ''}"
                            ${!paper.downloaded ? 'disabled' : ''}
                            data-paper-index="${index}"
                            id="preview-btn-${index}">
                        <span class="flex items-center justify-center space-x-2">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path>
                            </svg>
                            <span>Preview</span>
                        </span>
                    </button>

                    <button onclick="paperWhispererApp.openArxivLink('${paper.arxiv_url || '#'}')"
                            class="px-4 py-2.5 btn-secondary rounded-lg font-medium text-sm transition-all duration-200">
                        <span class="flex items-center justify-center space-x-2">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                            </svg>
                            <span>ArXiv</span>
                        </span>
                    </button>

                    <button onclick="paperWhispererApp.downloadPaper(${index})"
                            class="px-4 py-2.5 btn-secondary rounded-lg font-medium text-sm transition-all duration-200">
                        <span class="flex items-center justify-center space-x-2">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            <span>Download</span>
                        </span>
                    </button>
                </div>
            </div>
        `;
        
        return card;
    }


    async openPaperPreview(paperIndex) {
        const paper = this.currentPapers[paperIndex];
        if (!paper || !paper.downloaded) return;

        // Try multiple ways to find the preview button
        let previewButton = document.querySelector(`#preview-btn-${paperIndex}`);
        if (!previewButton) {
            previewButton = document.querySelector(`[data-paper-index="${paperIndex}"]`);
        }
        
        console.log('Preview button found:', previewButton); // Debug log
        
        if (!previewButton) {
            console.error('Preview button not found for index:', paperIndex);
            return;
        }
        
        const originalContent = previewButton.innerHTML;
        console.log('Original button content:', originalContent); // Debug log
        
        // Show professional spinner with processing message
        previewButton.innerHTML = `
            <span class="flex items-center justify-center space-x-2">
                <div class="small-spinner"></div>
                <span class="font-semibold">Processing...</span>
            </span>
        `;
        previewButton.disabled = true;
        previewButton.classList.add('opacity-75');
        
        console.log('Button spinner applied'); // Debug log

        try {
            this.currentPaper = paper;
            this.currentPage = 1;

            // Show main loading state as well
            this.showLoading('Processing PDF preview...');
            console.log('Main loading state shown'); // Debug log

            // Get paper pages info
            const response = await fetch(`${this.apiBase}/paper/${paperIndex}/info`);
            const data = await response.json();
            
            if (data.success) {
                this.totalPages = data.total_pages;
                this.setupPageSelector();
                await this.loadPage(1);
                
                document.getElementById('modalPaperTitle').textContent = paper.title;
                document.getElementById('paperModal').classList.remove('hidden');
                this.showSuccess('PDF preview loaded successfully');
            } else {
                this.showError(data.message || 'Preview loading failed');
            }
        } catch (error) {
            console.error('Error loading paper preview:', error);
            this.showError('Failed to process PDF preview');
        } finally {
            // Hide main loading state
            this.hideLoading();
            
            // Restore button state
            previewButton.innerHTML = originalContent;
            previewButton.disabled = false;
            previewButton.classList.remove('opacity-75');
            
            console.log('Button state restored'); // Debug log
        }
    }

    setupPageSelector() {
        const selector = document.getElementById('pageSelector');
        selector.innerHTML = '';
        
        for (let i = 1; i <= this.totalPages; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `Page ${i}`;
            selector.appendChild(option);
        }
        
        selector.value = this.currentPage;
    }

    async loadPage(pageNum) {
        if (!this.currentPaper) return;

        // Show loading state for page loading
        const pageImage = document.getElementById('pageImage');
        if (pageImage) {
            pageImage.style.opacity = '0.5';
        }

        try {
            const paperIndex = this.currentPapers.indexOf(this.currentPaper);
            const response = await fetch(`${this.apiBase}/paper/${paperIndex}/page/${pageNum}`);
            
            if (response.ok) {
                // Handle direct image response
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                
                document.getElementById('pageImage').src = imageUrl;
                document.getElementById('pageImage').style.opacity = '1';
                this.currentPage = pageNum;
                document.getElementById('pageSelector').value = pageNum;
                
                // Update navigation buttons
                document.getElementById('prevPageBtn').disabled = pageNum === 1;
                document.getElementById('nextPageBtn').disabled = pageNum === this.totalPages;
                
                // Hide previous analysis results
                document.getElementById('analysisResults').classList.add('hidden');
            } else {
                throw new Error('Failed to load page');
            }
        } catch (error) {
            console.error('Error loading page:', error);
            this.showError('Failed to load page');
            if (pageImage) {
                pageImage.style.opacity = '1';
            }
        }
    }

    navigatePage(direction) {
        const newPage = this.currentPage + direction;
        if (newPage >= 1 && newPage <= this.totalPages) {
            this.loadPage(newPage);
        }
    }

    goToPage(pageNum) {
        if (pageNum >= 1 && pageNum <= this.totalPages) {
            this.loadPage(pageNum);
        }
    }

    async analyzePage() {
        if (!this.currentPaper) return;

        const button = document.getElementById('analyzePageBtn');
        this.addButtonSpinner(button, 'Analyzing...');

        try {
            const paperIndex = this.currentPapers.indexOf(this.currentPaper);
            const response = await fetch(`${this.apiBase}/paper/${paperIndex}/analyze/${this.currentPage}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayAnalysisResults(result.analysis);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Page analysis failed');
        } finally {
            this.removeButtonSpinner(button, 'Analyze Page');
        }
    }

    displayAnalysisResults(analysis) {
        // Update content type
        document.getElementById('contentType').textContent = analysis.content_type || 'Unknown';
        
        // Update has diagram
        document.getElementById('hasDiagram').textContent = analysis.has_diagram ? 'Yes' : 'No';
        
        // Update explanation
        document.getElementById('explanation').textContent = analysis.explanation || 'No detailed analysis available';
        
        // Update insights
        const insightsContainer = document.getElementById('insights');
        insightsContainer.innerHTML = '';
        
        if (analysis.insights && analysis.insights.length > 0) {
            analysis.insights.forEach((insight, index) => {
                const insightDiv = document.createElement('div');
                insightDiv.className = 'bg-electric-50 border border-electric-200 rounded-xl p-4 text-sm text-dark-700';
                insightDiv.innerHTML = `<span class="text-electric-600 font-bold">#${index + 1}</span> ${insight}`;
                insightsContainer.appendChild(insightDiv);
            });
        } else {
            insightsContainer.innerHTML = '<div class="text-dark-500 text-sm">No insights available</div>';
        }
        
        // Update diagram analysis
        const diagramCard = document.getElementById('diagramAnalysisCard');
        if (analysis.diagram_analysis) {
            document.getElementById('diagramAnalysis').textContent = analysis.diagram_analysis;
            diagramCard.classList.remove('hidden');
        } else {
            diagramCard.classList.add('hidden');
        }
        
        // Update technical elements
        const technicalCard = document.getElementById('technicalElementsCard');
        if (analysis.technical_elements) {
            document.getElementById('technicalElements').textContent = analysis.technical_elements;
            technicalCard.classList.remove('hidden');
        } else {
            technicalCard.classList.add('hidden');
        }
        
        // Show results
        document.getElementById('analysisResults').classList.remove('hidden');
    }

    async clearData() {
        if (!confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
            return;
        }

        const button = document.getElementById('clearDataBtn');
        this.addButtonSpinner(button, 'Clearing...');

        try {
            const response = await fetch(`${this.apiBase}/clear`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentPapers = [];
                this.displayPapers();
                this.showSuccess('Data cleared successfully');
            } else {
                throw new Error(result.error || 'Clear failed');
            }
        } catch (error) {
            console.error('Clear error:', error);
            this.showError('Failed to clear data');
        } finally {
            this.removeButtonSpinner(button, 'Clear Data');
        }
    }

    // Button spinner utilities
    addButtonSpinner(button, loadingText) {
        if (button.dataset.loading === 'true') return;
        
        button.dataset.loading = 'true';
        button.dataset.originalText = button.innerHTML;
        button.disabled = true;
        
        button.innerHTML = `
            <span class="flex items-center justify-center space-x-3">
                <div class="button-spinner"></div>
                <span class="font-semibold">${loadingText}</span>
                <span class="text-xs opacity-75">Processing</span>
            </span>
        `;
    }

    removeButtonSpinner(button, originalText) {
        if (button.dataset.loading !== 'true') return;
        
        button.dataset.loading = 'false';
        button.disabled = false;
        
        if (button.dataset.originalText) {
            button.innerHTML = button.dataset.originalText;
        } else {
            button.innerHTML = `<span class="flex items-center justify-center space-x-3"><span>${originalText}</span></span>`;
        }
        
        delete button.dataset.originalText;
    }

    closePaperModal() {
        document.getElementById('paperModal').classList.add('hidden');
        this.currentPaper = null;
        this.currentPage = 1;
    }

    openArxivLink(url) {
        if (url && url !== '#') {
            window.open(url, '_blank');
        }
    }

    downloadPaper(paperIndex) {
        const paper = this.currentPapers[paperIndex];
        if (!paper) return;
        
        // For now, open ArXiv link - you can implement actual download later
        this.openArxivLink(paper.arxiv_url);
        this.showSuccess('Opening ArXiv page for download...');
    }

    async clearData() {
        if (!confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
            return;
        }

        const button = document.getElementById('clearDataBtn');
        this.addButtonSpinner(button, 'Clearing...');

        try {
            const response = await fetch(`${this.apiBase}/clear`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentPapers = [];
                this.displayPapers();
                this.showSuccess('Data cleared successfully');
            } else {
                throw new Error(result.error || 'Clear failed');
            }
        } catch (error) {
            console.error('Clear error:', error);
            this.showError('Failed to clear data');
        } finally {
            this.removeButtonSpinner(button, 'Clear Data');
        }
    }

    async loadExistingPapers() {
        try {
            const response = await fetch(`${this.apiBase}/papers`);
            const result = await response.json();
            
            if (result.success) {
                this.currentPapers = result.papers || [];
                this.displayPapers();
            }
        } catch (error) {
            console.error('Error loading existing papers:', error);
        }
    }

    showLoading(message) {
        this.isLoading = true;
        document.getElementById('loadingMessage').textContent = message;
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('progressText').textContent = '0/0 (0%)';
        document.getElementById('loadingState').classList.remove('hidden');
    }

    hideLoading() {
        this.isLoading = false;
        document.getElementById('loadingState').classList.add('hidden');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 z-50 max-w-sm w-full ${
            type === 'success' ? 'notification-success' : 'notification-error'
        } rounded-lg shadow-lg p-4 transition-all transform translate-x-full`;
        
        notification.innerHTML = `
            <div class="flex items-center">
                <div class="flex-shrink-0">
                    ${type === 'success' 
                        ? '<svg class="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>'
                        : '<svg class="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>'
                    }
                </div>
                <div class="ml-3 flex-1">
                    <p class="text-sm font-medium text-gray-800">${message}</p>
                    <p class="text-xs text-gray-500 mt-1 font-source">PaperWhisperer</p>
                </div>
                <div class="ml-4">
                    <button onclick="this.parentElement.parentElement.remove()" class="text-gray-400 hover:text-gray-600 transition-colors">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.parentElement.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    // Chat functionality methods
    openChatModal() {
        const chatModal = document.getElementById('chatModal');
        chatModal.classList.remove('hidden');
        
        // Set up chat event listeners
        this.setupChatEventListeners();
        
        // Load chat status
        this.loadChatStatus();
    }

    closeChatModal() {
        const chatModal = document.getElementById('chatModal');
        chatModal.classList.add('hidden');
        
        // Remove event listeners to prevent duplicates
        this.removeChatEventListeners();
    }

    setupChatEventListeners() {
        const sendBtn = document.getElementById('sendChatBtn');
        const clearBtn = document.getElementById('clearChatBtn');
        const chatInput = document.getElementById('chatInput');

        // Remove any existing listeners
        this.removeChatEventListeners();

        // Send message
        this.chatSendHandler = () => this.sendChatMessage();
        sendBtn.addEventListener('click', this.chatSendHandler);

        // Clear chat
        this.chatClearHandler = () => this.clearChat();
        clearBtn.addEventListener('click', this.chatClearHandler);

        // Enter key to send
        this.chatKeyHandler = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendChatMessage();
            }
        };
        chatInput.addEventListener('keydown', this.chatKeyHandler);
    }

    removeChatEventListeners() {
        const sendBtn = document.getElementById('sendChatBtn');
        const clearBtn = document.getElementById('clearChatBtn');
        const chatInput = document.getElementById('chatInput');

        if (this.chatSendHandler) {
            sendBtn.removeEventListener('click', this.chatSendHandler);
        }
        if (this.chatClearHandler) {
            clearBtn.removeEventListener('click', this.chatClearHandler);
        }
        if (this.chatKeyHandler) {
            chatInput.removeEventListener('keydown', this.chatKeyHandler);
        }
    }

    async loadChatStatus() {
        try {
            const response = await fetch(`${this.apiBase}/rag/status`);
            const data = await response.json();
            
            if (data.success && data.status) {
                const status = data.status;
                console.log('RAG Status:', status);
                
                // Update chat interface based on status
                if (!status.is_ready) {
                    this.addChatMessage('system', 'Chat service is initializing... Please wait a moment.');
                } else if (status.total_chunks === 0) {
                    this.addChatMessage('system', 'No papers are available for chat yet. Please add some papers and I\'ll process them automatically.');
                } else {
                    this.addChatMessage('system', `Ready to chat! I have processed ${status.processed_papers} papers with ${status.total_chunks} text segments.`);
                }
            }
        } catch (error) {
            console.error('Error loading chat status:', error);
            this.addChatMessage('system', 'Chat service is temporarily unavailable.');
        }
    }

    async sendChatMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput.value.trim();
        
        if (!message) return;

        // Clear input
        chatInput.value = '';

        // Add user message to chat
        this.addChatMessage('user', message);

        // Show loading
        this.showChatLoading();

        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // Hide loading
            this.hideChatLoading();

            if (data.success) {
                // Add AI response
                this.addChatMessage('ai', data.message, data.sources);
            } else {
                this.addChatMessage('system', data.message || 'Sorry, I encountered an error processing your question.');
            }

        } catch (error) {
            console.error('Error sending chat message:', error);
            this.hideChatLoading();
            this.addChatMessage('system', 'Network error. Please check your connection and try again.');
        }
    }

    addChatMessage(type, message, sources = null) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        
        if (type === 'user') {
            messageDiv.className = 'flex items-start space-x-3 justify-end';
            messageDiv.innerHTML = `
                <div class="flex-1 max-w-xs">
                    <div class="bg-blue-500 text-white rounded-lg p-4 shadow-sm">
                        <p class="text-sm">${this.escapeHtml(message)}</p>
                    </div>
                    <div class="text-xs text-gray-500 mt-1 text-right">You</div>
                </div>
                <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/>
                    </svg>
                </div>
            `;
        } else {
            const isSystem = type === 'system';
            const bgColor = isSystem ? 'bg-yellow-50 border border-yellow-200' : 'bg-white border';
            const iconColor = isSystem ? 'bg-yellow-500' : 'bg-blue-500';
            const label = isSystem ? 'System' : 'AI Assistant';
            
            messageDiv.className = 'flex items-start space-x-3';
            messageDiv.innerHTML = `
                <div class="w-8 h-8 ${iconColor} rounded-full flex items-center justify-center flex-shrink-0">
                    <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="${isSystem ? 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5l-6.928-12c-.77-.833-2.734-.833-3.464 0l-6.928 12c-.77.833.192 2.5 1.732 2.5z' : 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z'}"/>
                    </svg>
                </div>
                <div class="flex-1">
                    <div class="${bgColor} rounded-lg p-4 shadow-sm">
                        <div class="text-sm text-gray-700 leading-relaxed">${this.formatChatMessage(message)}</div>
                        ${sources ? this.formatSources(sources) : ''}
                    </div>
                    <div class="text-xs text-gray-500 mt-1">${label}</div>
                </div>
            `;
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    formatChatMessage(message) {
        // Basic formatting for chat messages
        return this.escapeHtml(message)
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    formatSources(sources) {
        if (!sources || sources.length === 0) return '';
        
        const sourcesList = sources.map(source => 
            `<span class="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full mr-1 mb-1">
                ${this.escapeHtml(source.title.substring(0, 30))}... (p.${source.page})
            </span>`
        ).join('');
        
        return `<div class="mt-3 pt-3 border-t border-gray-200">
            <div class="text-xs text-gray-600 mb-2">Sources:</div>
            <div>${sourcesList}</div>
        </div>`;
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    showChatLoading() {
        const chatStatus = document.getElementById('chatStatus');
        chatStatus.classList.remove('hidden');
    }

    hideChatLoading() {
        const chatStatus = document.getElementById('chatStatus');
        chatStatus.classList.add('hidden');
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        // Keep only the first message (the welcome message)
        const firstMessage = chatMessages.firstElementChild;
        chatMessages.innerHTML = '';
        if (firstMessage) {
            chatMessages.appendChild(firstMessage);
        }
        
        // Reload status
        this.loadChatStatus();
    }
}

// Global functions
function closePaperModal() {
    paperWhispererApp.closePaperModal();
}

function closeChatModal() {
    paperWhispererApp.closeChatModal();
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.paperWhispererApp = new PaperWhispererApp();
});
