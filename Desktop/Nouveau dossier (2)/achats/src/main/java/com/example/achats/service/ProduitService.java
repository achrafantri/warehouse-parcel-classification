package com.example.achats.service;

import com.example.achats.model.Categorie;
import com.example.achats.model.Produit;
import com.example.achats.repository.CategorieRepository;
import com.example.achats.repository.ProduitRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.Optional;

/**
 * Service li yconnecti controller bel repository
 */
@Service
public class ProduitService {

    private final ProduitRepository produitRepository;
    private final CategorieRepository categorieRepository;

    /**
     * Constructeur mta3 ProduitService
     */
    public ProduitService(ProduitRepository produitRepository, CategorieRepository categorieRepository) {
        this.produitRepository = produitRepository;
        this.categorieRepository = categorieRepository;
    }

    /**
     * tjib les produits mte3 kol page
     */
    public Page<Produit> getAllProduits(Pageable pageable) {
        return produitRepository.findAll(pageable);
    }

    /**
     * tjib produit par id
     */
    public Optional<Produit> getProduitById(Long id) {
        return produitRepository.findById(id);
    }

    /**
     * tasna3 produit jdid
     */
    public Produit createProduit(Produit produit) {
        return produitRepository.save(produit);
    }

    /**
     * modefier produit, fil code hedha l'id catégorie tnajjem barka tbedelha
     */
    public Produit updateProduit(Long id, Produit produitDetails) {
        if (produitDetails == null) {
            throw new IllegalArgumentException("Produit details cannot be null");
        }

        Produit produit = produitRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Produit non trouvé"));

        if (produitDetails.getCategorie() != null && produitDetails.getCategorie().getId() != null) {
            Categorie categorie = categorieRepository.findById(produitDetails.getCategorie().getId())
                    .orElseThrow(() -> new RuntimeException("Catégorie non trouvée"));
            produit.setCategorie(categorie);
        }

        produit.setNom(produitDetails.getNom());
        produit.setCode(produitDetails.getCode());
        produit.setDesignation(produitDetails.getDesignation());
        produit.setPrixUnitaire(produitDetails.getPrixUnitaire());
        produit.setPhoto(produitDetails.getPhoto());
        produit.setFournisseur(produitDetails.getFournisseur());
        produit.setNumfacture(produitDetails.getNumfacture());
        produit.setNumservice(produitDetails.getNumservice());

        return produitRepository.save(produit);
    }

    /**
     * Supprimer produit
     */
    public void deleteProduit(Long id) {
        produitRepository.deleteById(id);
    }

    /**
     * Filtrer par catégorie
     */
public Page<Produit> getProduitsByCategorie(Long categorieId, Pageable pageable) {
    try {
        return produitRepository.findByCategorieId(categorieId, pageable);
    } catch (Exception e) {
        // Log the exception (you may want to use a logger here)
        System.err.println("Error fetching products by category: " + e.getMessage());
        throw new RuntimeException("Unable to fetch products for the specified category.");
    }
}
}
